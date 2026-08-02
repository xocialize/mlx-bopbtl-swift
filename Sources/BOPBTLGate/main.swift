// bopbtl-gate — the parity gate: MLX-Swift forward vs the torch fp32 fixture logits.
// usage: bopbtl-gate <weights.safetensors> <fixtures.safetensors>
import BOPBTLMLXCore
import Foundation
import MLX
import MLXRandom

setvbuf(stdout, nil, _IONBF, 0)

let args = CommandLine.arguments

// --gpu-probe: re-test the Metal conv window that forced the CPU pin (run on every mlx-swift bump;
// remove the pin in ScratchDetector when this reports clean).
if args.count == 2, args[1] == "--gpu-probe" {
    MLXRandom.seed(3)
    var dirty = false
    for (cin, cout, hw) in [(128, 128, 130), (128, 128, 66), (256, 256, 66)] {
        let x = MLXRandom.uniform(low: -1.0, high: 1.0, [1, hw, hw, cin])
        let w = MLXRandom.uniform(low: -0.1, high: 0.1, [cout, 3, 3, cin])
        let gpu = conv2d(x, w, stride: [1, 1], padding: [0, 0])
        let cpu = conv2d(x, w, stride: [1, 1], padding: [0, 0], stream: .device(.cpu))
        eval(gpu, cpu)
        let d = (gpu - cpu).abs().max().item(Float.self)
        let bad = d > 1e-3
        dirty = dirty || bad
        print(String(format: "in%d out%d @%d: max|gpu-cpu| %.6f %@", cin, cout, hw, d, bad ? "⚠️ STILL DIVERGES" : "✅ clean"))
    }
    print(dirty ? "\nWINDOW STILL DIRTY — keep the CPU pin" : "\nWINDOW CLEAN — remove the CPU pin in ScratchDetector.logits")
    exit(dirty ? 1 : 0)
}
guard args.count == 3 else { print("usage: bopbtl-gate <weights.safetensors> <fixtures.safetensors> | --gpu-probe"); exit(2) }

if ProcessInfo.processInfo.environment["BOPBTL_CPU"] != nil {
    Device.setDefault(device: Device(.cpu))
    print("(cpu stream)")
}
let detector = try ScratchDetector(weightsURL: URL(fileURLWithPath: args[1]))

// Stage bisect: bopbtl-gate <weights> <fixtures> against a stages file via BOPBTL_STAGES env.
if let stagesPath = ProcessInfo.processInfo.environment["BOPBTL_STAGES"] {
    let stages = try MLX.loadArrays(url: URL(fileURLWithPath: stagesPath))
    let fixtures0 = try MLX.loadArrays(url: URL(fileURLWithPath: args[2]))
    let x = fixtures0["a256_in"]!.asType(.float32).transposed(0, 2, 3, 1)
    for (name, mlxOut) in detector.stagedForward(x, subStages: true) {
        guard let ref = stages[name] else { continue }
        let got = mlxOut.transposed(0, 3, 1, 2)
        eval(got)
        let d = (got - ref.asType(.float32)).abs()
        print(String(format: "  %@ max|Δ| %.6f  mean %.6f", name, d.max().item(Float.self), d.mean().item(Float.self)))
    }
    exit(0)
}
let fixtures = try MLX.loadArrays(url: URL(fileURLWithPath: args[2]))

var worstDB = Double.infinity
var failures = 0
for name in ["a256", "b256x384", "c272x256"] {
    guard let xin = fixtures["\(name)_in"], let ref = fixtures["\(name)_out"] else {
        print("❌ fixture \(name) missing"); failures += 1; continue
    }
    // Fixtures are torch NCHW [1,1,H,W] → NHWC.
    let x = xin.asType(.float32).transposed(0, 2, 3, 1)
    let y = detector.logits(x)                              // [1,H,W,1]
    let out = y.transposed(0, 3, 1, 2)                      // back to NCHW for comparison
    eval(out)
    let delta = (out - ref.asType(.float32)).abs()
    let maxDelta = delta.max().item(Float.self)
    let refPower = (ref.asType(.float32) * ref.asType(.float32)).mean().item(Float.self)
    let mse = (delta * delta).mean().item(Float.self)
    let db = 10 * log10(Double(refPower) / Double(max(mse, .leastNormalMagnitude)))
    worstDB = min(worstDB, db)
    let pass = db > 60
    if !pass { failures += 1 }
    print(String(format: "%@ %@: max|Δ| %.5f  SNR %.2f dB", pass ? "✅" : "❌", name, maxDelta, db))
}
print(failures == 0 ? String(format: "\nGATE PASS (worst %.2f dB)", worstDB)
                    : "\nGATE FAIL (\(failures))")
exit(failures == 0 ? 0 : 1)
