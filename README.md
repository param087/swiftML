# swiftML

Swift library for Machine Learning based on [Swift for TensorFlow](https://github.com/tensorflow/swift) Project.


## Documentation
 [https://param087.github.io/swiftML/](https://param087.github.io/swiftML/ )



## Getting Started

 * **Install locally**: [Swift for TensorFlow toolchain](https://github.com/tensorflow/swift/blob/master/Installation.md) and Jupyter Kernel for Swift for TensorFlow [swift-jupyter](https://github.com/google/swift-jupyter).
 
 


* **Google Colaboratory**: The fastest way to get started is to try out right in your browser. Just open up a [tutorial](https://github.com/param087/swiftML/tree/master/Notebooks), or start from a [blank notebook](https://colab.research.google.com/github/tensorflow/swift/blob/master/notebooks/blank_swift.ipynb)!


### How to include the library in your package

Add the library to your projects dependencies in the Package.swift file as shown below.
```swift 
dependencies: [
        .package(url: "https://github.com/param087/swiftML", .exact("0.0.2")),
    ],
```

### How to include the library in your Jupyter Notebook using `%install` directives.

`%install` directives let you install SwiftPM packages so that your notebook
can import them:

```swift

// Install the swiftML package from GitHub.
%install '.package(url: "https://github.com/param087/swiftML", from: "0.0.2")' swiftML

// Install the swiftML package that's in the local directory.
%install '.package(path: "$cwd/swiftML")' swiftML
```

## Contributing

We welcome contribution from everyone. Read the [contribution guide](https://github.com/param087/CONTRIBUTION.md) for information on of how to get started.

## Community

swiftML discussions happen on the [Slack](https://join.slack.com/t/swiftml/shared_invite/enQtODgwMjEzOTIxOTkwLWMzYzlmZTQwNjJkNzBiNzNhZGZmN2FhZjBlNDgxNjVjMDkxNDRlM2UxYmMzMmE4ZTMzZmE0ODIxZGQ2NzdiYWI).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.