import TensorFlow

/// A linear regression model
///
/// Linear regression with support for gradient descent and singular vector decomposition (svd).
public class LinearRegression {

    var iterations: Int
    var learningRate: Float
    var gradientDescent: Bool
    var fitIntercept: Bool
    public var weights: Tensor<Float>

    /// Creates a linear regression model.
    ///
    /// - Parameters
    ///   - gradientDescent: The gradient descent or singular vector decomposition method to be
    ///     used for learning, default to `false`.
    ///   - iterations: The number of iterations for gradient descent, default to `1000`.
    ///   - learningRate: The learning rate for gardient descent, default to `0.001`.
    ///   - fitIntercept: whether to calculate the intercept for this model. If set to False, no
    ///     intercept will be used in calculations, default set to `true`.
    public init(
        gradientDescent: Bool = false,
        iterations: Int = 1000,
        learningRate: Float = 0.001,
        fitIntercept: Bool = true
    ) {
        precondition(iterations > 0, "Number of iterations must be greater than zero.")
        precondition(learningRate >= 0, "Learning Rate must be non-negative.")

        self.iterations = iterations
        self.learningRate = learningRate
        self.gradientDescent = gradientDescent
        self.weights = Tensor<Float>(0)
        self.fitIntercept = fitIntercept
    }

    /// Initialize weights between [-1/N, 1/N].
    /// - Parameters featuresCount: The number of features in training data.
    internal func initializeWeights(featuresCount: Int) {
    
        var w = Tensor<Float>(zeros: [featuresCount, 1])
        let limit: Float = 1 / sqrt(Float(featuresCount))
    
        for i in 0..<featuresCount {
            w[i] = Tensor<Float>([Float.random(in: -limit...limit)])
        }
        // assign weights with randomly initialize weights.
        self.weights = w
    }
    
    /// Fit linear model.
    ///
    /// - Parameters
    ///   - data: Training data Tensor<Float> of shape [number of samples, number of features].
    ///   - labels: Target value Tensor<Float> of shape [number of samples, 1].
    public func fit(data: Tensor<Float>, labels: Tensor<Float>) {
        
        precondition(data.shape[0] == labels.shape[0],
            "data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "data must be non-empty.")
        precondition(data.shape[1] >= 1, "data must have atleast single feature.")
        precondition(data.shape[0] > 0, "labels must be non-empty.")
        precondition(labels.shape[1] == 1, "labels must have single target feature.")

        var modifiedData: Tensor<Float> = data
       
        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            modifiedData = ones.concatenated(with: data, alongAxis: -1)
        }

        if self.gradientDescent == false {       
            let svd = Raw.svd(matmul(modifiedData.transposed(), modifiedData))
            let s = Raw.diag(diagonal: svd.s)
            let dataSqRegInv = matmul(matmul(svd.v, matrixPseudoInverse(s)), svd.u.transposed())
            self.weights = matmul(matmul(dataSqRegInv, modifiedData.transposed()), labels)
        } else {
            self.initializeWeights(featuresCount: modifiedData.shape[1])

            for _ in 0..<self.iterations {
                let predictedLabels = matmul(modifiedData, self.weights)
                let weightsGradient =
                    -1 * matmul(modifiedData.transposed(), (labels - predictedLabels))
                self.weights = self.weights - (self.learningRate * weightsGradient)
            }
        }
    }

    /// Returns predict using linear model.
    ///
    /// - Parameter data: Sample data tensor of shape [number of samples, number of features].
    /// - Returns: Predicted value tensor.
    public func prediction(for data: Tensor<Float>) -> Tensor<Float> {
    
        precondition(data.shape[0] > 0, "data must be non-empty.")

        var modifiedData: Tensor<Float> = data
       
        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            modifiedData = ones.concatenated(with: data, alongAxis: -1)
        }

        let predictedLabels = matmul(modifiedData, self.weights)
        return predictedLabels
    }

    /// Returns the coefficient of determination R^2 of the prediction.
    ///
    /// - Parameters
    ///   - data: Sample tensor of shape [number of samples, number of features].
    ///   - labels: Target value tensor of shape [number of samples, 1].
    /// - Returns: The coefficient of determination R^2 of the prediction.
    public func score(data: Tensor<Float>, labels: Tensor<Float>) -> Float {

        precondition(data.shape[0] == labels.shape[0],
            "data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "data must be non-empty.")
        precondition(labels.shape[0] > 0, "labels must be non-empty.")
        
        let predictedLabels = self.prediction(for: data)
        let u = pow((labels - predictedLabels),2).sum()
        let v = pow((labels - labels.mean()), 2).sum()
        let score = (1 - (u/v))
        
        return Float(score.scalarized())
    }
}