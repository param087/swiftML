# swiftML

Swift library for Machine Learning.

### Requirements

 * [Swift for TensorFlow](https://github.com/tensorflow/swift)
 * [swift-jupyter](https://github.com/google/swift-jupyter)

### or

* [Google Colaboratory](https://colab.research.google.com/github/tensorflow/swift/blob/master/notebooks/blank_swift.ipynb)

---
## %install directives

`%install` directives let you install SwiftPM packages so that your notebook
can import them:

```swift

// Install the swiftML package from GitHub.
%install '.package(url: "https://github.com/param087/swiftML", from: "0.0.1")' swiftML

// Install the swiftML package that's in the local directory.
%install '.package(path: "path to swiftML")' swiftML
```
