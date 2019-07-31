import TensorFlow

/// A linear regression model.
///
/// Linear regression with support for gradient descent and singular vector decomposition (svd).
public class LinearRegression {
    /// The gradient descent or singular vector decomposition method to be used for learning.
    var gradientDescent: Bool
    /// The number of iterations for gradient descent.
    var iterationCount: Int
    /// The linear rate gradient descent.
    var learningRate: Float
    /// Whether to calculate the intercept for this model.
    var fitIntercept: Bool
    /// The weights of the model.
    public var weights: Tensor<Float>

    /// Creates a linear regression model.
    ///
    /// - Parameters:
    ///   - gradientDescent: The gradient descent or singular vector decomposition method to be
    ///     used for learning. The default is `false`.
    ///   - iterationCount: The number of iterations for gradient descent. The default is `1000`.
    ///   - learningRate: The learning rate for gradient descent. The default is `0.001`.
    ///   - fitIntercept: Whether to calculate the intercept for this model. If `false`, no
    ///     intercept will be used in calculations. The default is `true`.
    public init(
        gradientDescent: Bool = false,
        iterationCount: Int = 1000,
        learningRate: Float = 0.001,
        fitIntercept: Bool = true
    ) {
        precondition(iterationCount > 0, "Iteration count must be positive.")
        precondition(learningRate >= 0, "Learning rate must be non-negative.")

        self.gradientDescent = gradientDescent
        self.iterationCount = iterationCount
        self.learningRate = learningRate
        self.weights = Tensor<Float>(0)
        self.fitIntercept = fitIntercept
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
    ///   - labels: Target value with shape `[sample count, 1]`.
    public func fit(data: Tensor<Float>, labels: Tensor<Float>) {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(data.shape[1] >= 1, "Data must have feature count greater than one.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")
        precondition(labels.shape[1] == 1, "Labels must have single target feature.")
        precondition(data.shape[0] == labels.shape[0],
                     "Data and labels must have the same sample count.")

        var data: Tensor<Float> = data
        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            data = ones.concatenated(with: data, alongAxis: -1)
        }
        if self.gradientDescent == false {       
            let svd = Raw.svd(matmul(data.transposed(), data))
            let s = Raw.diag(diagonal: svd.s)
            let dataSqRegInv = matmul(matmul(svd.v, s.pseudoinverse), svd.u.transposed())
            self.weights = matmul(matmul(dataSqRegInv, data.transposed()), labels)
        } else {
            self.initializeWeights(featuresCount: data.shape[1])

            for _ in 0..<self.iterationCount {
                let predictedLabels = matmul(data, self.weights)
                let weightsGradient = -1 * matmul(data.transposed(), (labels - predictedLabels))
                self.weights = self.weights - (self.learningRate * weightsGradient)
            }
        }
    }

    /// Returns prediction using linear model.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    /// - Returns: Predicted output.
    public func prediction(for data: Tensor<Float>) -> Tensor<Float> {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")

        var data: Tensor<Float> = data
        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            data = ones.concatenated(with: data, alongAxis: -1)
        }
        let predictedLabels = matmul(data, self.weights)
        return predictedLabels
    }

    /// Returns the coefficient of determination (`R^2`) of the prediction.
    ///
    /// - Parameters
    ///   - data: Sample data with shape `[sample count, feature count]`.
    ///   - labels: Target value with shape `[sample count, 1]`.
    /// - Returns: The coefficient of determination (`R^2`) of the prediction.
    public func score(data: Tensor<Float>, labels: Tensor<Float>) -> Float {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")
        precondition(data.shape[0] == labels.shape[0],
                     "Data and labels must have the same sample count.")
        
        let predictedLabels = self.prediction(for: data)
        let u = pow((labels - predictedLabels),2).sum()
        let v = pow((labels - labels.mean()), 2).sum()
        let score = (1 - (u/v))
        
        return Float(score.scalarized())
    }
}
