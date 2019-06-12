import TensorFlow

/// A linear regression model
///
/// Linear regression with support for gradient descent and singular vector decomposition( svd).
public class LinearRegression {

    public var iterations: Int
    public var learningRate: Float
    public var weights: Tensor<Float>
    public var gradientDescent: Bool

    /// Creates a linear regression model.
    ///
    /// - Parameters
    ///   - gradientDescent: The gradient descent or singular vector decomposition method to be
    ///     used for learning,
    ///     default to 'false'.   
    ///   - iterations: The number of iterations for gradient descent, default to 1000.
    ///   - learningRate: The learning rate for gardient descent, default to 0.001
    public init(
        gradientDescent: Bool = false,
        iterations: Int = 1000,
        learningRate: Float = 0.001
    ) {
        precondition(iterations > 0, "Number of iterations must be greater than zero.")
        precondition(learningRate >= 0, "Learning Rate must be non-negative.")

        self.iterations = iterations
        self.learningRate = learningRate
        self.gradientDescent = gradientDescent
        self.weights = Tensor<Float>(0)
    }

    /// Initialize weights between [-1/N, 1/N].
    /// - Parameters nFeatures: The number of features in training.
    public func initializeWeights(nFeatures: Int) {
    
        var w = Tensor<Float>(zeros:[nFeatures, 1])
        let limit: Float = 1 / sqrt(Float(nFeatures))
    
        for i in 0..<nFeatures {
            w[i] = Tensor<Float>([Float.random(in: -limit...limit)])
        }
        // assign weights with randomly initialize weights.
        self.weights = w
    }
    
    /// Fit linear model.
    ///
    /// - Parameters
    ///   - X: Training data Tensor<Flaot> of shape [number of samples, number of features].
    ///   - y: Target value Tensor<Flaot> of shape [number of samples, 1].
    public func fit(X: Tensor<Float>, y: Tensor<Float>) {
        
        precondition(X.shape[0] == y.shape[0], "X and y must have same number of samples.")
        precondition(X.shape[0] > 0, "X must be non-empty.")
        precondition(X.shape[1] >= 1, "X must have atleast single feature.")
        precondition(y.shape[0] > 0, "y must be non-empty.")
        precondition(y.shape[1] == 1, "y must have single target.")

        let ones = Tensor<Float>(ones: [X.shape[0], 1])
        let x = ones.concatenated(with: X, alongAxis: -1)

        if self.gradientDescent == false {
           
            let svd = Raw.svd(matmul(x.transposed(), x))
            let s = Raw.diag(diagonal: svd.s)
            let X_sq_reg_inv = matmul(matmul(svd.v, pinv(s)), svd.u.transposed())
            self.weights = matmul(matmul(X_sq_reg_inv, x.transposed()), y)

        } else {

            self.initializeWeights(nFeatures: x.shape[1])

            for _ in 0..<self.iterations {
        
                let y_pred = matmul(x, self.weights)
                let grad_w = -1 * matmul(x.transposed(), (y - y_pred))
                self.weights = self.weights - (self.learningRate * grad_w)
            }
        }
    }

    /// Returns predict using linear model.
    ///
    /// - Parameter X: Smaple data tensor of shape [number of samples, number of features].
    /// - Returns: Predicted value tensor.
    public func predict(X: Tensor<Float>) -> Tensor<Float> {
    
        precondition(X.shape[0] > 0, "X must be non-empty.")

        let ones = Tensor<Float>(ones: [X.shape[0], 1])
        let x = ones.concatenated(with: X, alongAxis: -1)
        let y_pred = matmul(x, self.weights)
        return y_pred
    }

    /// Returns the coefficient of determination R^2 of the prediction.
    ///
    /// - Parameters
    ///   - X: Sample tensor of shape [number of samples, number of features].
    ///   - y: Target value tensor of shape [number of samples, 1].
    /// - Returns: The coefficient of determination R^2 of the prediction.
    public func score(X: Tensor<Float>, y: Tensor<Float>) -> Float {

        precondition(X.shape[0] == y.shape[0], "X and y must have same number of samples.")
        precondition(X.shape[0] > 0, "X must be non-empty.")
        precondition(y.shape[0] > 0, "y must be non-empty.")
        
        let y_pred = self.predict(X: X)
        let u = pow((y - y_pred),2).sum() 
        let v = pow((y - y.mean()), 2).sum()
        let score = (1 - (u/v))
        
        return Float(score.scalarized())
    }
}