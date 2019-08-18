import TensorFlow

/// Gradient Descent linear regression.
///
/// Reference: ["linear regression using gradient descent"](
/// http://cs229.stanford.edu/notes/cs229-notes1.pdf)
public class GradientDescentLinearRegression: LinearRegression {
    /// The number of iterations for gradient descent.
    var iterationCount: Int
    /// The linear rate gradient descent.
    var learningRate: Float
    /// Whether to calculate the intercept for this model.
    public var fitIntercept: Bool
    /// The weights of the model.
    public var weights: Tensor<Float>

    /// Creates a linear regression model.
    ///
    /// - Parameters:
    ///   - iterationCount: The number of iterations for gradient descent. The default is `1000`.
    ///   - learningRate: The learning rate for gradient descent. The default is `0.001`.
    ///   - fitIntercept: Whether to calculate the intercept for this model. If `false`, no
    ///     intercept will be used in calculations. The default is `true`.
    public init(
        iterationCount: Int = 1000,
        learningRate: Float = 0.001,
        fitIntercept: Bool = true
    ) {
        precondition(iterationCount > 0, "Iteration count must be positive.")
        precondition(learningRate >= 0, "Learning rate must be non-negative.")
        self.iterationCount = iterationCount
        self.learningRate = learningRate
        self.fitIntercept = fitIntercept
        self.weights = Tensor<Float>(0)

    }

    /// Initialize weights between `[-1/N, 1/N]`.
    /// - Parameter featuresCount: The number of features in training data.
    internal func initializeWeights(featuresCount: Int) {
        // Randomly initialize weights.
        var w = Tensor<Float>(zeros: [featuresCount, 1])
        let limit: Float = 1 / sqrt(Float(featuresCount))
        for i in 0..<featuresCount {
            w[i] = Tensor<Float>([Float.random(in: -limit...limit)])
        }
        self.weights = w
    }

    /// Fit a linear model.
    ///
    /// - Parameters:
    ///   - data: Training data with shape `[sample count, feature count]`.
    ///   - labels: Target value with shape `[sample count, target count]`.
    public func fit(data: Tensor<Float>, labels: Tensor<Float>) {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(data.shape[1] >= 1,
            "Data must have feature count greater than or equal to one.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")
        precondition(labels.shape[1] >= 1,
            "Labels must have target feature count greater than or equal to one.")
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have the same sample count.")

        var data: Tensor<Float> = data
        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            data = ones.concatenated(with: data, alongAxis: -1)
        }
        
        self.initializeWeights(featuresCount: data.shape[1])

        for _ in 0..<self.iterationCount {
            let predictedLabels = matmul(data, self.weights)
            let weightsGradient = -1 * matmul(data.transposed(), (labels - predictedLabels))
            self.weights = self.weights - (self.learningRate * weightsGradient)
        }
    }
}