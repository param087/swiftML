import TensorFlow

/// Gaussian naive bayes classifier.
///
/// Gaussian naive bayes classifier used to classify continuous features.
public class GaussianNB {

    // Unique classes in target value set.
    public var cls: Tensor<Int32>
    // Tensor contains the index of class in cls.
    public var idx: Tensor<Int32>
    // The mean and the standard deviation of each attribute for each class.
    public var model: Tensor<Float>
  
    /// Create a Gaussian naive model.
    public init() {
        self.cls = Tensor<Int32>([0])
        self.idx = Tensor<Int32>([0])
        self.model = Tensor<Float>([0.0])
    }

    /// Fit Gaussian naive bayes classifier model.
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
        
        var separated = [[Tensor<Float>]]()
        self.model = Tensor<Float>(zeros:[self.cls.shape[0], X.shape[1], 2])

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

        // Calculate the mean and the standard deviation of each attribute for each class.
        for i in 0..<separated.count {
            let classSet = Tensor<Float>(Tensor<Float>(separated[i]))
            let mean = classSet.mean(alongAxes: 0)
            let std = classSet.standardDeviation(alongAxes: 0)
            self.model[i] = mean.reshaped(to: [X.shape[1], 1])
                .concatenated(with: std.reshaped(to: [X.shape[1], 1]), alongAxis: -1) 
        }
    }

    /// Returns gaussian distribution in log.
    ///
    /// - Parameters
    ///   - x: Input tensor to find gausssian distribution.
    ///   - mean: Mean of input tensor.
    ///   - std: Standard deviation of input tensor.
    /// - Returns: Ggaussian distribution in log.
    public func prob(x: Tensor<Float>, mean: Tensor<Float>, std: Tensor<Float>) -> Tensor<Float> {

        // Tensor Double is used to prevent tenor overflow.
        let xd = Tensor<Double>(x)
        let meand = Tensor<Double>(mean)
        let stdd = Tensor<Double>(std)
        let exponent = exp(-1.0 * (pow((xd - meand), 2) / (2.0 * pow(stdd, 2))))
        return Tensor<Float>(log(exponent / (pow(Tensor<Double>(2.0 * Double.pi) , 0.5) * stdd)))
    }

    /// Returns predict log probability.
    ///
    /// - Parameter X: Input tensor to predict log probability.
    /// - Return: predict log probability
    public func predictLogProba(X: Tensor<Float>) -> Tensor<Float> {

        var predictLogProb = Tensor<Float>(zeros:[X.shape[0], self.cls.shape[0]])
 
        for i in 0..<X.shape[0] {
            for j in 0..<self.model.shape[0] {
                var sum = Tensor<Float>(0.0)
                for k in 0..<self.model[j].shape[0] {
                    sum = sum + 
                        self.prob(x: X[i][k], mean: self.model[j][k][0], std: self.model[j][k][1])
                }
                predictLogProb[i][j] = sum
            }
        }
        return predictLogProb
    }
  
    /// Returns classified test tensor.
    ///
    /// - Parameter X: Test Tensor<Float> of shape [number of samples, number of features].
    /// - Returns: classified tensor.
    public func predict(X: Tensor<Float>) -> Tensor<Float> {

        precondition(X.shape[0] > 0, "X must be non-empty.")

        var labels = Tensor<Int32>(zeros:[X.shape[0]])
        var predLogProb = self.predictLogProba(X: X)
        
        for i in 0..<X.shape[0] {
            labels[i] = self.cls[Int(predLogProb[i].argmax().scalarized())]
        }
        return Tensor<Float>(labels)
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

        let result = self.predict(X: X)
        var count: Int = 0
        for i in 0..<result.shape[0] {
            if result[i] == y[i] {
                count = count + 1
            }
        }
        let score:Float = Float(count)/Float(y.shape[0])
        return score
    }
}