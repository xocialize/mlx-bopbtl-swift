import CoreGraphics
import Foundation
import Hub
import ImageIO
import MLX
import MLXProfiling
import MLXToolKit
import BOPBTLMLXCore
import UniformTypeIdentifiers

public enum BOPBTLPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case maskEncodeFailed
    case weightsMissing(String)
}

/// Configuration for the BOPBTL scratch detector.
public struct BOPBTLConfiguration: PackageConfiguration, WeightSourcing, Sendable, Codable {
    /// Default binarization threshold when the request doesn't supply one. Upstream's is 0.4.
    public var defaultThreshold: Float
    /// Explicit weights file (bypasses download) — parity/dev use.
    public var weightsURL: URL?
    /// Engine-stamped model store root (`ModelStorable`).
    public var modelsRootDirectory: URL?

    public static let repo = "mlx-community/BOPBTL-scratch-detection-fp32-mlx"

    public init(defaultThreshold: Float = 0.4,
                weightsURL: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.defaultThreshold = defaultThreshold
        self.weightsURL = weightsURL
        self.modelsRootDirectory = modelsRootDirectory
    }

    public var weightSources: [WeightSource] {
        [WeightSource(role: "weights", repo: Self.repo, revision: nil,
                      matching: ["model.safetensors"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if weightsURL != nil { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    // Persist the portable knob only; stamped roots and absolute paths re-stamp per session.
    private enum CodingKeys: String, CodingKey { case defaultThreshold }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(defaultThreshold: try c.decodeIfPresent(Float.self, forKey: .defaultThreshold) ?? 0.4)
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(defaultThreshold, forKey: .defaultThreshold)
    }
}

/// The MLXEngine `defectDetect` package over the BOPBTL scratch-detection UNet — the first backer
/// of the capability (contract 1.31.0). Photo in → damage mask out; the canonical consumer is the
/// registered `imageInpaint` (detect → mask → fill, no hand-painted brush).
///
/// ⚠️ **Inference runs on the CPU stream by the CORE's own pin** — the Metal fp32 conv2d kernel is
/// wrong in this model's geometry window (receipt in `BOPBTLMLXCore.ScratchDetector`). Detection
/// runs at short-side 256, so CPU costs nothing perceptible (~1.0 s/image measured through this
/// package). `requiredBackends` still declares `.metalGPU` — the engine's placement vocabulary has
/// no CPU pool, the working set is trivial either way, and the pin is the core's internal concern.
///
/// Preprocessing mirrors `Global/detection.py` `scale_256`: BT.601 grayscale, BICUBIC-class resize
/// to short side 256 with both dims rounded to multiples of 16, Normalize([0.5],[0.5]).
/// ⚠️ CoreGraphics high-quality resampling is NOT bit-identical to PIL BICUBIC (the fleet's
/// documented PIL trap) — numeric parity is gated at the CORE on exact tensors; the package-level
/// tolerance is semantic (a mask), where the resampler difference is immaterial.
@InferenceActor
public final class BOPBTLDetectPackage: ModelPackage {
    public typealias Configuration = BOPBTLConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // MIT both layers: upstream LICENSE + headers + README §License (which extends MIT to
            // "the codes and the pretrained model" verbatim). The README's "academic research use
            // only" line was reviewed against those formal grants and cleared 2026-08-01
            // (PORT-QUEUE.md P7 records the question and the resolution).
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: BOPBTLConfiguration.repo, revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // MEASURED through this package surface (phys_footprint sampler, 1080p photo →
                // scale_256 448×256 → forward → mask), stated WITH its cache configuration
                // (the REQUIREMENTS.md rule): at the engine's shipping 2 GiB cache cap the peak is
                // **3.56–3.91 GB**; at a 256 MB cap the true active set is **1.63 GB** — the
                // difference is MLX buffer-pool retention, and even the CPU stream's buffers live
                // in the unified pool the cap governs. ~1.0 s/image. Load is lazy (8 MB post-load;
                // the fp32 weights materialize on first forward). Per-stage evals in the core bound
                // the lazy graph (fully lazy measured 3.9 GB even at a 448×256 forward — every
                // conv's im2col temporaries materialized at once). Declared at the shipping-cache
                // basis with headroom.
                footprints: [QuantFootprint(quant: .fp32,
                                            residentBytes: 200_000_000,
                                            peakActivationBytes: 3_900_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                DefectDetectContract.descriptor(
                    name: "bopbtl-scratch-detect",
                    summary: "BOPBTL scratch/defect detection: photo in, damage mask out "
                        + "(binary at a 0…1 threshold, default 0.4, or raw probability)."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var detector: ScratchDetector?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard detector == nil else { return }
        let url: URL
        if let explicit = configuration.weightsURL {
            guard FileManager.default.fileExists(atPath: explicit.path) else {
                throw BOPBTLPackageError.weightsMissing(explicit.path)
            }
            url = explicit
        } else {
            let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
            let dir = try await hub.snapshot(from: Hub.Repo(id: BOPBTLConfiguration.repo),
                                             matching: ["model.safetensors"]) { progress, speed in
                WeightDownloadProgress.report(fraction: progress.fractionCompleted, bytesPerSecond: speed)
            }
            url = dir.appendingPathComponent("model.safetensors")
        }
        detector = try ScratchDetector(weightsURL: url)
    }

    public func unload() async {
        detector = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()   // CAN-1: first act of run()
        guard let detector else { throw PackageError.notLoaded }
        guard request.capability == .defectDetect,
              let req = request as? DefectDetectRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        guard let source = CGImageSourceCreateWithData(req.image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw BOPBTLPackageError.imageDecodeFailed("unreadable \(req.image.format.rawValue) data")
        }
        let srcW = cg.width, srcH = cg.height

        // scale_256: short side → 256, both dims rounded to /16.
        let (mw, mh) = Self.scale256Dims(width: srcW, height: srcH)
        guard let gray = Self.grayPlane(cg, width: mw, height: mh) else {
            throw BOPBTLPackageError.imageDecodeFailed("grayscale resample (\(mw)x\(mh))")
        }

        try Task.checkCancellation()   // pre-forward
        let prof = MLXProfiler.shared
        prof.beginRun("bopbtl defectDetect \(srcW)x\(srcH) → \(mw)x\(mh)")
        // [-1,1] NHWC [1,H,W,1]; the core pins its own CPU stream (kernel-window receipt).
        let x = (MLXArray(gray, [1, mh, mw, 1]) * (2.0 / 255.0)) - 1.0
        let probs = detector.probabilities(x)
        eval(probs)
        prof.endRun(denominators: ["image": 1])

        try Task.checkCancellation()   // post-forward

        let kind = req.preferredKind
        let threshold = kind == .binary ? (req.threshold ?? configuration.defaultThreshold) : nil
        var plane = probs[0, 0..., 0..., 0]
        if let t = threshold {
            plane = (plane .>= t).asType(.float32)
        }
        eval(plane)

        // Nearest-neighbour back to source dims — the reference's `F.interpolate(mode="nearest")`.
        guard let mask = Self.maskPNG(plane: plane, modelW: mw, modelH: mh,
                                      outW: srcW, outH: srcH) else {
            throw BOPBTLPackageError.maskEncodeFailed
        }
        return DefectDetectResponse(
            mask: Matte(data: mask, width: srcW, height: srcH, kind: kind),
            appliedThreshold: threshold)
    }

    // MARK: - Preprocessing / encoding

    nonisolated static func scale256Dims(width: Int, height: Int) -> (Int, Int) {
        var ow = Double(width), oh = Double(height)
        if width < height { oh = oh / ow * 256; ow = 256 } else { ow = ow / oh * 256; oh = 256 }
        let w = Int((ow / 16).rounded()) * 16
        let h = Int((oh / 16).rounded()) * 16
        return (max(w, 16), max(h, 16))
    }

    /// BT.601 grayscale at the model size, high-quality resample.
    nonisolated static func grayPlane(_ cg: CGImage, width: Int, height: Int) -> [Float]? {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: &rgba, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        var out = [Float](repeating: 0, count: width * height)
        for i in 0 ..< width * height {
            out[i] = 0.299 * Float(rgba[i * 4]) + 0.587 * Float(rgba[i * 4 + 1])
                   + 0.114 * Float(rgba[i * 4 + 2])
        }
        return out
    }

    /// Grayscale PNG of the mask plane, nearest-resampled to the output dims.
    nonisolated static func maskPNG(plane: MLXArray, modelW: Int, modelH: Int,
                                    outW: Int, outH: Int) -> Data? {
        let values = plane.asArray(Float.self)
        var gray = [UInt8](repeating: 0, count: modelW * modelH)
        for i in 0 ..< values.count {
            gray[i] = UInt8(max(0, min(255, (values[i] * 255).rounded())))
        }
        guard let provider = CGDataProvider(data: Data(gray) as CFData),
              let small = CGImage(width: modelW, height: modelH, bitsPerComponent: 8, bitsPerPixel: 8,
                                  bytesPerRow: modelW,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent)
        else { return nil }
        guard let ctx = CGContext(data: nil, width: outW, height: outH, bitsPerComponent: 8,
                                  bytesPerRow: outW, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return nil }
        ctx.interpolationQuality = .none          // nearest — a mask must not be smoothed
        ctx.draw(small, in: CGRect(x: 0, y: 0, width: outW, height: outH))
        guard let big = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(dest, big, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }
}

extension BOPBTLDetectPackage {
    public nonisolated static var registration: PackageRegistration {
        .of(BOPBTLDetectPackage.self)
    }
}
