import TensorFlow 

/// Bernoulli naive bayes classifier.
///
/// Bernoulli naive bayes classifier used to classify discrete binary features.
public class BernoulliNB {

    public var alpha: Float
    // The prior log probability for each class.
    public var classLogPrior: Tensor<Float>
    // Log probability of each word.
    public var featureLogProb: Tensor<Float>
    // Unique classes in target value set.
    public var cls: Tensor<Int32>
    // Tensor contains the index of class in cls.
    public var idx: Tensor<Int32>

    /// Create a bernoulli naive model.
    ///
    /// - Parameter alpha: Additive smoothing parameter, default to 1.0.
    public init(
        alpha: Float = 1.0
    ) {
        self.alpha = alpha
        self.cls = Tensor<Int32>([0])
        self.idx = Tensor<Int32>([0])
        self.classLogPrior = Tensor<Float>([0.0])
        self.featureLogProb = Tensor<Float>([0.0])
    }
  
    /// Fit bernoulli naive bayes classifier model.
    ///
    /// - Parameters
    ///   - X: Training data Tensor<Float> of shape [number of samples, number of features].
    ///   - y: Target value Tensor<Float> of shape [number of samples].
    public func fit(X: Tensor<Float>, y: Tensor<Float>) {

        precondition(X.shape[0] == y.shape[0], "X and y must have same number of samples.")
        precondition(X.shape[0] > 0, "X must be non-empty.")
        precondition(X.shape[1] >= 1, "X must have atleast single feature.")
        precondition(y.shape[0] > 0, "y must be non-empty.")

        let labels = Tensor<Int32>(y)
        // find unique classes in target values.
        (self.cls, self.idx) = Raw.unique(labels.flattened())

        precondition(self.cls.shape[0] == 2, "y must have only two classes.")

        // Initialize the classLogPrior and featureLogProb based on number of features and sample.
        var separated = [[Tensor<Float>]]()
        self.classLogPrior = Tensor<Float>(zeros:[self.cls.shape[0]])
        self.featureLogProb = Tensor<Float>(zeros:[self.cls.shape[0], X.shape[1]])

        // Seperate the samples based on classes.
        for cind in 0..<self.cls.shape[0] {
            var classSet = [Tensor<Float>]()

            for xind in 0..<labels.shape[0] {
                if self.cls[cind] == labels[xind] {
                    classSet.append(X[xind])
                }
            }
            separated.append(classSet)
        }

        let sampleCount = X.shape[0]

        // Calculate logp^(c) , the prior log probability for each class.
        for ind in 0..<separated.count {
            let temp = Float(separated[ind].count)/Float(sampleCount)
            self.classLogPrior[ind] = Tensor<Float>(log(temp))
        }

        var count = Tensor<Float>(zeros:[separated.count, X.shape[1]])

        // Count each word for each class and add alpha as smoothing.
        for i in 0..<separated.count {
            for j in 0..<X.shape[1] {
                var temp = Tensor<Float>(0.0)
                for k in 0..<separated[i].count {
                  temp = temp + separated[i][k][j]
                }
                count[i][j] = temp + self.alpha
            }
        }
      
        let smoothing = 2.0 * self.alpha

        for i in 0..<separated.count {
            // Size is the number of documents in each class + smoothing,
            // where smoothing is 2 * self.alpha.
            var size = Tensor<Float>(0.0)
            size = Tensor<Float>(Float(separated[i].count) + smoothing)

            // Probability of each word.
            for j in 0..<count[i].shape[0] {
                self.featureLogProb[i][j] = count[i][j]/size
            }    
        }
    }

    /// Returns log-probability estimates for the test tensor.
    ///
    /// - Parameter X: Text Tensor<Float> of shape [number of samples, number of features].
    public func predictLogProba(X: Tensor<Float>) -> Tensor<Float> {

        var predictLogProb = Tensor<Float>(zeros:[X.shape[0], self.cls.shape[0]])
        
        for i in 0..<X.shape[0] {
            var temp = log(self.featureLogProb) * X[i]
            temp =  temp + log(1.0 - self.featureLogProb) * abs(X[i] - 1.0)
            predictLogProb[i] = temp.sum(alongAxes: 1).flattened() + self.classLogPrior
        }
        return predictLogProb
    }

    /// Returns classified test tensor.
    ///
    /// - Parameter X: Test Tensor<Float> of shape [number of samples, number of features].
    /// - Returns: classified tensor.
    public func predict(X: Tensor<Float>) -> Tensor<Int32> {

        precondition(X.shape[0] > 0, "X must be non-empty.")
         
        let predictLogProb = self.predictLogProba(X: X)

        var prediction = Tensor<Int32>(zeros: [X.shape[0]])
        for i in 0..<X.shape[0] {
            prediction[i] = self.cls[Int(predictLogProb[i].argmax().scalarized())]
        }
        return prediction
    }

    /// Returns mean accuracy on the given test data and labels.
    ///
    /// - Parameters
    ///   - X: Sample tensor of shape [number of samples, number of features].
    ///   - y: Target label tensor of shape [number of samples].
    /// - Returns: Returns the mean accuracy on the given test data and labels.
    public func score(X: Tensor<Float>, y: Tensor<Float>) -> Float {

        precondition(X.shape[0] == y.shape[0], "X and y must have same number of samples.")
        precondition(X.shape[0] > 0, "X must be non-empty.")
        precondition(y.shape[0] > 0, "y must be non-empty.")

        let result = Tensor<Float>(self.predict(X: X))
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