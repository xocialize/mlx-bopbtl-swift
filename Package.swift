// swift-tools-version: 6.2
import PackageDescription

// mlx-bopbtl-swift — the BOPBTL scratch DETECTOR (Bringing Old Photos Back to Life, the detection
// UNet only) for MLXEngine's `defectDetect` capability. Upstream:
// microsoft/Bringing-Old-Photos-Back-to-Life — MIT, with the LICENSE file, every source header,
// and README §License extending MIT to the pretrained model verbatim; cleared for porting
// 2026-08-01 after review (PORT-QUEUE.md P7 records the licence question and its resolution).
//
// ⚠️ The detector only. The repo's restoration generators are NOT ported — we ship our own
// restore fleet; this model supplies the missing DETECTION half (mask in → the registered
// `imageInpaint` fixes what it marks).
let package = Package(
    name: "mlx-bopbtl-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "BOPBTLMLXCore", targets: ["BOPBTLMLXCore"]),
        .library(name: "MLXBOPBTL", targets: ["MLXBOPBTL"]),
        .executable(name: "bopbtl-gate", targets: ["BOPBTLGate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.43.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
    ],
    targets: [
        .target(name: "BOPBTLMLXCore", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
        ]),
        .target(name: "MLXBOPBTL", dependencies: ["BOPBTLMLXCore",
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "Hub", package: "swift-transformers"),
            .product(name: "MLXProfiling", package: "mlx-profiling"),
            .product(name: "MLX", package: "mlx-swift"),
        ], swiftSettings: [.swiftLanguageMode(.v5)]),   // fleet convention for engine targets (Hub closure sendability)
        .executableTarget(name: "BOPBTLGate", dependencies: ["BOPBTLMLXCore",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXRandom", package: "mlx-swift"),
        ]),
        .testTarget(name: "BOPBTLMLXCoreTests", dependencies: ["BOPBTLMLXCore",
            .product(name: "MLX", package: "mlx-swift"),
        ]),
        .testTarget(name: "MLXBOPBTLTests", dependencies: ["MLXBOPBTL",
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
        ]),
    ]
)
