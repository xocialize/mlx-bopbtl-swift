// Weightless invariants + the env-gated live parity gate.
//
// ⚠️ XCTest, NOT swift-testing — deliberately. Under `--build-system swiftbuild` the Cmlx metallib
// is located via Bundle lookup relative to the .xctest bundle; swift-testing suites execute in the
// runner process (whose Bundle.main is the runner, not the bundle), so ANY MLX call aborts with
// "Failed to load the default metallib" even though the metallib is present. XCTest hosts inside
// the bundle and resolves it. New variant of the metallib CI trap — recorded in the fleet memory.
import Foundation
import MLX
import XCTest

@testable import BOPBTLMLXCore

final class ReflectionPadTests: XCTestCase {
    override func invokeTest() { withMLXCPU { super.invokeTest() } }

    /// Pinned against `nn.ReflectionPad2d` semantics: mirror WITHOUT repeating the edge.
    func testMatchesTorchSemantics() {
        let padded = reflectionPad2d(MLXArray([Float]([0, 1, 2, 3, 4, 5, 6, 7]), [1, 2, 4, 1]), 1)
        eval(padded)
        XCTAssertEqual(padded.dim(1), 4)
        XCTAssertEqual(padded.dim(2), 6)
        // Padded row 1 = original row 0 reflected: [1,0,1,2,3,2]
        XCTAssertEqual(padded[0, 1, 0..., 0].asArray(Float.self), [1, 0, 1, 2, 3, 2])
        // Padded row 0 = original row 1 (mirror above the top edge).
        XCTAssertEqual(padded[0, 0, 0..., 0].asArray(Float.self), [5, 4, 5, 6, 7, 6])
    }

    func testPad3OnSmallGrid() {
        let x = MLXArray((0 ..< 25).map(Float.init), [1, 5, 5, 1])
        let padded = reflectionPad2d(x, 3)
        eval(padded)
        XCTAssertEqual(padded.dim(1), 11)
        XCTAssertEqual(padded.dim(2), 11)
        XCTAssertEqual(padded[0, 0, 0, 0].item(Float.self), 18)   // reflect(3,3) → source (3,3)
        XCTAssertEqual(padded[0, 5, 5, 0].item(Float.self), 12)   // centre untouched
    }
}

final class LiveParityTests: XCTestCase {
    override func invokeTest() { withMLXCPU { super.invokeTest() } }

    /// The full fixture gate as a test — same 60 dB bar as `bopbtl-gate`. Skips without weights.
    func testFixturesMatchTorch() throws {
        let env = ProcessInfo.processInfo.environment
        guard let weights = env["BOPBTL_WEIGHTS"], let fixturesPath = env["BOPBTL_FIXTURES"] else {
            throw XCTSkip("set BOPBTL_WEIGHTS and BOPBTL_FIXTURES to run live parity")
        }
        let detector = try ScratchDetector(weightsURL: URL(fileURLWithPath: weights))
        let fixtures = try MLX.loadArrays(url: URL(fileURLWithPath: fixturesPath))
        for name in ["a256", "b256x384", "c272x256"] {
            let x = fixtures["\(name)_in"]!.asType(.float32).transposed(0, 2, 3, 1)
            let out = detector.logits(x).transposed(0, 3, 1, 2)
            eval(out)
            let ref = fixtures["\(name)_out"]!.asType(.float32)
            let mse = ((out - ref) * (out - ref)).mean().item(Float.self)
            let power = (ref * ref).mean().item(Float.self)
            let db = 10 * log10(Double(power) / Double(max(mse, .leastNormalMagnitude)))
            XCTAssertGreaterThan(db, 60, "\(name): \(db) dB")
        }
    }
}
