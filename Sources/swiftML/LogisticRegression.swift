import TensorFlow

/// A Logistic regression classifier.
///
/// In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme.
public class LogisticRegression {

    public var learningRate: Float
    public var iterations: Int
    public var w = [Tensor<Float>]()
    // The tensor of unique classes
    public var cls: Tensor<Int32>
    // The tensor contains the index to unique classes
    public var idx: Tensor<Int32>
  
    /// Creates a logistic regression model.
    ///
    /// - Parameters
    ///   - iterations: The number of iterations for gradient descent, default to 0.1.
    ///   - learningRate: The learning rate for gardient descent, default to 1000.
    public init(
        learningRate: Float = 0.1,
        iterations: Int = 1000
    ) {

        precondition(iterations > 0, "Number of iterations must be greater than zero.")
        precondition(learningRate >= 0, "Learning Rate must be non-negative.")

        self.learningRate = learningRate
        self.iterations = iterations
        self.cls = Tensor<Int32>(0)
        self.idx = Tensor<Int32>(0)
    }
    
    /// Fit logistic regression model.
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
    
        let ones = Tensor<Float>(ones:[X.shape[0], 1])
        let x = ones.concatenated(with: X, alongAxis: -1)
        let m = Float(X.shape[0])

        let labels = Tensor<Int32>(y)
        (self.cls, self.idx) = Raw.unique(labels.flattened())

        precondition(self.cls.shape[0] >= 2,"labels must contains atleast 2 classes")
        
        // loop through each class, uses the one-vs-rest (OvR) scheme. 
        for i in 0..<self.cls.shape[0] {

            let condition = Raw.equal(labels, cls[i])
            let t = Tensor<Int32>(ones:[labels.shape[0],1])
            let e1 = Tensor<Int32>(zeros:[labels.shape[0],1])

            // create temparary label for one-vs-rest scheme, based on class the selected class
            // labeled as one while rest as zeros.
            var y_temp = Tensor<Float>(Raw.select(condition: condition, t: t, e: e1))
            y_temp = y_temp.reshaped(to: [y_temp.shape[0], 1])
    
            // weights of selected class in one-vs-rests scheme.
            var t_w = Tensor<Float>(ones:[x.shape[1], 1])

            for _ in 0..<self.iterations {
                let output = matmul(x, t_w)
                let errors = y_temp - sigmoid(output)
                t_w = t_w + ((self.learningRate / m) * matmul(x.transposed(), errors))
            }

            self.w.append(t_w)
        }
    }
    
    /// Return the prediction of single sample
    ///
    /// - Parameters X: Sample tuple Tensor<Flaot> of shape [1, number of features].
    /// - Returns: Predicted class label.
    public func predictOne(X: Tensor<Float>) -> Tensor<Float> {
        var output = Tensor<Float>(zeros:[w.count, 1])
        var counter: Int = 0

        for t_w in self.w {
            output[counter] = matmul(X.reshaped(to: [1, X.shape[0]]), t_w)[0]
            counter = counter + 1
        }

        let index: Int = Int(output.argmax().scalarized())
        let cl: Tensor<Int32> = self.cls[index]

        return Tensor<Float>([Tensor<Float>(cl)])
    }
    
    /// Returns predict using logistic regression classifier.
    ///
    /// - Parameter X: Smaple data tensor of shape [number of samples, number of features].
    /// - Returns: Predicted target tensor of class labels.
    public func predict(X: Tensor<Float>) -> Tensor<Float> {

        precondition(X.shape[0] > 0, "X must be non-empty.")

        let ones = Tensor<Float>(ones:[X.shape[0], 1])
        let x = ones.concatenated(with: X, alongAxis: -1)
        
        var result = Tensor<Float>(zeros:[x.shape[0], 1])
        for i in 0..<x.shape[0] {
            result[i] = self.predictOne(X: x[i])
        }
        return result
    }

    /// Returns mean accuracy on the given test data and labels.
    ///
    /// - Parameters
    ///   - X: Sample tensor of shape [number of samples, number of features].
    ///   - y: Target label tensor of shape [number of samples, 1].
    /// - Returns: Returns the mean accuracy on the given test data and labels.
    public func score(X: Tensor<Float>, y: Tensor<Float>) -> Float {

        precondition(X.shape[0] == y.shape[0], "X and y must have same number of samples.")
        precondition(X.shape[0] > 0, "X must be non-empty.")
        precondition(y.shape[0] > 0, "y must be non-empty.")

        let result = self.predict(X: X)
        var count: Int = 0
        for i in 0..<result.shape[0] {
            if result[i] == y[i] {
                count = count + 1
            }
        }
        let score: Float = Float(count)/Float(y.shape[0])
        return score
    }
}