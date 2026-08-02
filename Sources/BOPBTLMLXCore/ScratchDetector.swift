// ScratchDetector.swift — BOPBTLMLXCore
//
// The BOPBTL detection UNet (Global/detection_models/networks.py `UNet`, config
// in=1 out=1 depth=4 conv_num=2 wf=6 padding batch_norm up_mode="upsample" antialiasing=True),
// 37,626,177 params, transcribed against the released `FT_Epoch_latest.pt` (`model_state`).
//
// 🔑 **Functional structs, not MLXNN Modules — deliberately.** This model carries 20 BatchNorms, and
// the fleet has already paid for `Module.training` defaulting to `true` (the C14 lesson: PROD
// BiRefNet silently used per-image stats, 68% off, invisible to eyeball). Here BatchNorm is an
// explicit per-channel affine computed from the checkpoint's running stats — there is no training
// flag to forget, the trap is eliminated by construction rather than by discipline.
//
// Layout: NHWC (MLX native). Torch conv weights [O,I,kH,kW] are transposed to [O,kH,kW,I] at load.
//
// ⚠️ Faithfulness notes against the reference:
// - ReflectionPad2d is manual (MLX has no reflect pad): flipped-edge slices concatenated, exact.
// - The antialiased Downsample (adobe/antialiased-cnns BlurPool) is reflect-pad(1) + a FIXED
//   depthwise [1,2,1]⊗[1,2,1]/16 kernel at stride 2. The kernel ships in the checkpoint as a
//   buffer (`down_sample.N.4.filt`) and is LOADED, not reconstructed — parity over cleverness.
// - Decoder upsampling is bilinear ×2 `align_corners=False` + reflect-pad + 3×3 conv.
// - `center_crop` of the bridge is implemented even though /16-aligned inputs make it a no-op —
//   the reference has it, and an input that isn't /16-aligned should crop, not crash.
// - The output is PRE-SIGMOID logits, exactly what `model(x)` returns upstream; sigmoid and the
//   0.4 threshold belong to the caller (they are product policy, not architecture).

import Foundation
import MLX
import MLXNN

public enum ScratchDetectorError: Error, CustomStringConvertible {
    case missingWeight(String)
    case badShape(String, expected: [Int], got: [Int])

    public var description: String {
        switch self {
        case .missingWeight(let k): return "checkpoint missing tensor: \(k)"
        case .badShape(let k, let e, let g): return "\(k): expected shape \(e), got \(g)"
        }
    }
}

/// Reflection padding on H and W of an NHWC tensor — `nn.ReflectionPad2d(p)`, exact: mirrors
/// WITHOUT repeating the edge sample. Implemented as one `take` per axis over a precomputed
/// reflected index vector (MLX has no flip/negative-stride slice).
func reflectionPad2d(_ x: MLXArray, _ p: Int) -> MLXArray {
    guard p > 0 else { return x }
    func reflectedIndices(_ extent: Int) -> MLXArray {
        var idx = [Int32]()
        idx.reserveCapacity(extent + 2 * p)
        for i in stride(from: p, through: 1, by: -1) { idx.append(Int32(i)) }
        idx.append(contentsOf: (0 ..< extent).map(Int32.init))
        for i in stride(from: extent - 2, through: extent - 1 - p, by: -1) { idx.append(Int32(i)) }
        return MLXArray(idx)
    }
    var out = x.take(reflectedIndices(x.dim(1)), axis: 1)
    out = out.take(reflectedIndices(x.dim(2)), axis: 2)
    return out
}

/// A conv whose BatchNorm (eval semantics) is carried as an explicit per-channel affine.
struct ConvBNLReLU {
    let weight: MLXArray          // [O, kH, kW, I]
    let bias: MLXArray            // [O]
    let bnScale: MLXArray?        // gamma / sqrt(var + eps)
    let bnShift: MLXArray?        // beta - mean * bnScale
    let pad: Int
    let leaky: Bool

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv2d(reflectionPad2d(x, pad), weight, stride: [1, 1], padding: [0, 0])
        out = out + bias
        if let s = bnScale, let b = bnShift { out = out * s + b }
        if leaky { out = leakyRelu(out, negativeSlope: 0.2) }
        return out
    }
}

/// BlurPool: reflect-pad 1 + fixed depthwise 3×3, stride 2 (`antialiasing.Downsample`, filt 3).
struct BlurPool {
    let filt: MLXArray            // [C, 3, 3, 1] depthwise
    let channels: Int

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2d(reflectionPad2d(x, 1), filt, stride: [2, 2], padding: [0, 0], groups: channels)
    }
}

public final class ScratchDetector {

    // first: ReflPad(3) + Conv7 + LReLU
    private let first: ConvBNLReLU
    // 4× (pre-downsample conv+BN+LReLU, then BlurPool), 4× double conv blocks
    private let downPre: [ConvBNLReLU]
    private let downPool: [BlurPool]
    private let downBlocks: [[ConvBNLReLU]]
    // 4× (bilinear ×2 + ReflPad + conv3 [no BN, no act], then double conv block)
    private let upConvs: [ConvBNLReLU]
    private let upBlocks: [[ConvBNLReLU]]
    private let last: ConvBNLReLU
    private let upsample = Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: false))

    /// Load from the converted fp32 safetensors (torch `state_dict` keys, NCHW conv layout).
    public init(weightsURL: URL) throws {
        let raw = try MLX.loadArrays(url: weightsURL)

        func tensor(_ key: String) throws -> MLXArray {
            guard let t = raw[key] else { throw ScratchDetectorError.missingWeight(key) }
            return t.asType(.float32)
        }
        /// Torch conv weight [O,I,kH,kW] → MLX [O,kH,kW,I].
        func convW(_ key: String) throws -> MLXArray {
            try tensor(key).transposed(0, 2, 3, 1)
        }
        /// BatchNorm running stats → the eval-mode affine. eps matches nn.BatchNorm2d (1e-5).
        func bn(_ prefix: String) throws -> (MLXArray, MLXArray) {
            let gamma = try tensor("\(prefix).weight"), beta = try tensor("\(prefix).bias")
            let mean = try tensor("\(prefix).running_mean"), variance = try tensor("\(prefix).running_var")
            let scale = gamma / sqrt(variance + 1e-5)
            return (scale, beta - mean * scale)
        }
        func convBN(_ conv: String, _ bnPrefix: String?, pad: Int, leaky: Bool = true) throws -> ConvBNLReLU {
            let (s, b): (MLXArray?, MLXArray?)
            if let bnPrefix {
                let affine = try bn(bnPrefix)
                (s, b) = (affine.0, affine.1)
            } else { (s, b) = (nil, nil) }
            return ConvBNLReLU(weight: try convW("\(conv).weight"), bias: try tensor("\(conv).bias"),
                               bnScale: s, bnShift: b, pad: pad, leaky: leaky)
        }

        first = try convBN("first.1", nil, pad: 3)

        var pres: [ConvBNLReLU] = [], pools: [BlurPool] = [], blocks: [[ConvBNLReLU]] = []
        for i in 0 ..< 4 {
            pres.append(try convBN("down_sample.\(i).1", "down_sample.\(i).2", pad: 1))
            // The BlurPool kernel is a checkpoint buffer: [C,1,3,3] depthwise → MLX [C,3,3,1].
            let filt = try tensor("down_sample.\(i).4.filt").transposed(0, 2, 3, 1)
            pools.append(BlurPool(filt: filt, channels: filt.dim(0)))
            blocks.append([try convBN("down_path.\(i).block.1", "down_path.\(i).block.2", pad: 1),
                           try convBN("down_path.\(i).block.5", "down_path.\(i).block.6", pad: 1)])
        }
        downPre = pres; downPool = pools; downBlocks = blocks

        var ups: [ConvBNLReLU] = [], ublocks: [[ConvBNLReLU]] = []
        for i in 0 ..< 4 {
            ups.append(try convBN("up_path.\(i).up.2", nil, pad: 1, leaky: false))
            ublocks.append([try convBN("up_path.\(i).conv_block.block.1", "up_path.\(i).conv_block.block.2", pad: 1),
                            try convBN("up_path.\(i).conv_block.block.5", "up_path.\(i).conv_block.block.6", pad: 1)])
        }
        upConvs = ups; upBlocks = ublocks

        last = try convBN("last.1", nil, pad: 1, leaky: false)
    }

    /// Forward: NHWC `[1, H, W, 1]` in `[-1, 1]` (grayscale, Normalize([0.5],[0.5])) →
    /// pre-sigmoid logits `[1, H, W, 1]`. H and W should be multiples of 16 (the reference
    /// resizes to guarantee it; non-multiples survive via the bridge crop but are off-reference).
    ///
    /// 🚨 **Runs on the CPU stream, scoped, on purpose — the Metal fp32 conv2d kernel is WRONG in
    /// this model's exact geometry window.** Measured 2026-08-01 (isolated random-data probe, this
    /// package's `convprobe` receipt): GPU-vs-CPU max|Δ| ≈ 0.077–0.117 whenever
    /// **inC ≥ 128 AND outC ≥ 128 AND spatial ≥ ~64** (in96→128, in128→64, @34: all exact at 1e-5;
    /// in128→128 @66/@130 and in256→256 @66: ~10 bits lost). Through the whole net that costs
    /// 51 dB vs the fp32 torch reference; the CPU stream scores **119.8 dB worst** and the entire
    /// 256-class forward is sub-second on CPU — detection resolution makes the GPU unnecessary.
    /// Same *family* as the NAX split-K GEMM window (mlx#3797/#3810): a shape-windowed Metal
    /// kernel defect, invisible outside its window. ⚠️ On every mlx-swift bump run
    /// `bopbtl-gate --gpu-probe`; when the window comes back clean, remove this pin.
    public func logits(_ input: MLXArray) -> MLXArray {
        Device.withDefaultDevice(Device(.cpu)) { logitsImpl(input) }
    }

    private func logitsImpl(_ input: MLXArray) -> MLXArray {
        var x = first(input)

        var bridges: [MLXArray] = []
        for i in 0 ..< 4 {
            bridges.append(x)
            x = downPool[i](downPre[i](x))
            for block in downBlocks[i] { x = block(x) }
            // ⚠️ Per-stage eval is a MEMORY bound, not a style choice: fully lazy, the CPU-stream
            // graph materializes every conv's im2col temporary simultaneously at the final eval —
            // measured 3.9 GB peak for a 448×256 forward. Evaluating per stage retires each
            // stage's temporaries before the next allocates; peak fell to the working set.
            eval(x)
        }

        for i in 0 ..< 4 {
            x = upConvs[i](upsample(x))
            var bridge = bridges[3 - i]
            // center_crop(bridge, up.shape) — a no-op on /16-aligned inputs, exact otherwise.
            let dy = (bridge.dim(1) - x.dim(1)) / 2, dx = (bridge.dim(2) - x.dim(2)) / 2
            if dy > 0 || dx > 0 {
                bridge = bridge[0..., dy ..< (dy + x.dim(1)), dx ..< (dx + x.dim(2)), 0...]
            }
            x = concatenated([x, bridge], axis: 3)
            for block in upBlocks[i] { x = block(x) }
            eval(x)   // same bound on the decoder side
        }
        return last(x)
    }

    /// Per-stage outputs for parity bisecting — mirrors the reference stage naming.
    public func stagedForward(_ input: MLXArray, subStages: Bool = false) -> [(String, MLXArray)] {
        var out: [(String, MLXArray)] = []
        var x = first(input)
        out.append(("first", x))
        var bridges: [MLXArray] = []
        for i in 0 ..< 4 {
            bridges.append(x)
            x = downPool[i](downPre[i](x))
            out.append(("downsample\(i)", x))
            if subStages && i == 0 {
                let b = downBlocks[0][0]
                let conv1 = conv2d(reflectionPad2d(x, b.pad), b.weight, stride: [1, 1], padding: [0, 0]) + b.bias
                out.append(("d0_conv1", conv1))
                let bn1 = conv1 * b.bnScale! + b.bnShift!
                out.append(("d0_bn1", bn1))
                let lrelu1 = leakyRelu(bn1, negativeSlope: 0.2)
                out.append(("d0_lrelu1", lrelu1))
                let b2 = downBlocks[0][1]
                let conv2 = conv2d(reflectionPad2d(lrelu1, b2.pad), b2.weight, stride: [1, 1], padding: [0, 0]) + b2.bias
                out.append(("d0_conv2", conv2))
                out.append(("d0_bn2", conv2 * b2.bnScale! + b2.bnShift!))
            }
            for block in downBlocks[i] { x = block(x) }
            out.append(("down\(i)", x))
        }
        for i in 0 ..< 4 {
            x = upConvs[i](upsample(x))
            var bridge = bridges[3 - i]
            let dy = (bridge.dim(1) - x.dim(1)) / 2, dx = (bridge.dim(2) - x.dim(2)) / 2
            if dy > 0 || dx > 0 {
                bridge = bridge[0..., dy ..< (dy + x.dim(1)), dx ..< (dx + x.dim(2)), 0...]
            }
            x = concatenated([x, bridge], axis: 3)
            for block in upBlocks[i] { x = block(x) }
            out.append(("up\(i)", x))
        }
        out.append(("last", last(x)))
        return out
    }

    /// Scratch probability map: `sigmoid(logits)`, same shape as the input.
    public func probabilities(_ input: MLXArray) -> MLXArray {
        sigmoid(logits(input))
    }
}
