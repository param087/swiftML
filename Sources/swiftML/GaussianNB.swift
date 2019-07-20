import TensorFlow

/// Gaussian naive bayes classifier.
///
/// Gaussian naive bayes classifier used to classify continuous features.
public class GaussianNB {

    // Unique classes in target value set.
    public var classes: Tensor<Int32>
    // Tensor contains the index of class in classes.
    public var indices: Tensor<Int32>
    // The mean and the standard deviation of each attribute for each class.
    public var model: Tensor<Float>
  
    /// Create a Gaussian naive model.
    public init() {
        self.classes = Tensor<Int32>([0])
        self.indices = Tensor<Int32>([0])
        self.model = Tensor<Float>([0.0])
    }

    /// Fit Gaussian naive bayes classifier model.
    ///
    /// - Parameters
    ///   - data: Training data Tensor<Float> of shape [number of samples, number of features].
    ///   - labels: Target value Tensor<Float> of shape [number of samples].
    public func fit(data: Tensor<Float>, labels: Tensor<Float>) {
      
        precondition(data.shape[0] == labels.shape[0], 
            "Data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "Data must be non-empty.")
        precondition(data.shape[1] >= 1, "Data must have atleast single feature.")
        precondition(labels.shape[0] > 0, "labels must be non-empty.")

        let labels = Tensor<Int32>(labels)
        // find unique classes in target values.
        (self.classes, self.indices) = Raw.unique(labels.flattened())
        
        var separated = [[Tensor<Float>]]()
        self.model = Tensor<Float>(zeros: [self.classes.shape[0], data.shape[1], 2])

        // Seperate the samples based on classes.
        for classIndex in 0..<self.classes.shape[0] {
            var classArray = [Tensor<Float>]()
            for labelIndex in 0..<labels.shape[0] {
                if self.classes[classIndex] == labels[labelIndex] {
                    classArray.append(data[labelIndex])
                }
            }
            separated.append(classArray)
        }

        // Calculate the mean and the standard deviation of each attribute for each class.
        for i in 0..<separated.count {
            let classArray = Tensor<Float>(Tensor<Float>(separated[i]))
            let mean = classArray.mean(alongAxes: 0)
            let std = classArray.standardDeviation(alongAxes: 0)
            self.model[i] = mean.reshaped(to: [data.shape[1], 1])
                .concatenated(with: std.reshaped(to: [data.shape[1], 1]), alongAxis: -1) 
        }
    }

    /// Returns gaussian distribution in log.
    ///
    /// - Parameters
    ///   - data: Input tensor to find gausssian distribution.
    ///   - mean: Mean of input tensor.
    ///   - std: Standard deviation of input tensor.
    /// - Returns: Ggaussian distribution in log.
    internal func prob(
        data: Tensor<Float>,
        mean: Tensor<Float>,
        std: Tensor<Float>
    ) -> Tensor<Float> {

        // Tensor Double is used to prevent tenor overflow.
        let data = Tensor<Double>(data)
        let mean = Tensor<Double>(mean)
        let std = Tensor<Double>(std)
        let squaredDiff = pow((data - mean), 2)
        let exponent = exp(-1.0 * ( squaredDiff / (2.0 * pow(std, 2))))
        return Tensor<Float>(log(exponent / (pow(Tensor<Double>(2.0 * Double.pi) , 0.5) * std)))
    }

    /// Returns predict log probability.
    ///
    /// - Parameter data: Input tensor to predict log probability.
    /// - Return: predict log probability
    public func predictLogProba(data: Tensor<Float>) -> Tensor<Float> {

        var predictLogProb = Tensor<Float>(zeros:[data.shape[0], self.classes.shape[0]])
 
        for i in 0..<data.shape[0] {
            for j in 0..<self.model.shape[0] {
                var sum = Tensor<Float>(0.0)
                for k in 0..<self.model[j].shape[0] {
                    sum = sum + 
                        self.prob(data: data[i][k], 
                            mean: self.model[j][k][0], std: self.model[j][k][1])
                }
                predictLogProb[i][j] = sum
            }
        }
        return predictLogProb
    }
  
    /// Returns classified test tensor.
    ///
    /// - Parameter data: Test Tensor<Float> of shape [number of samples, number of features].
    /// - Returns: classified tensor.
    public func prediction(for data: Tensor<Float>) -> Tensor<Float> {

        precondition(data.shape[0] > 0, "Data must be non-empty.")

        var labels = Tensor<Int32>(zeros: [data.shape[0]])
        let predLogProb = self.predictLogProba(data: data)
        
        for i in 0..<data.shape[0] {
            labels[i] = self.classes[Int(predLogProb[i].argmax().scalarized())]
        }
        return Tensor<Float>(labels)
    }

    /// Returns mean accuracy on the given test data and labels.
    ///
    /// - Parameters
    ///   - data: Sample tensor of shape [number of samples, number of features].
    ///   - labels: Target label tensor of shape [number of samples].
    /// - Returns: Returns the mean accuracy on the given test data and labels.
    public func score(data: Tensor<Float>, labels: Tensor<Float>) -> Float {

        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "data must be non-empty.")
        precondition(labels.shape[0] > 0, "labels must be non-empty.")

        let result = self.prediction(for: data)
        var count: Int = 0
        for i in 0..<result.shape[0] {
            if result[i] == labels[i] {
                count = count + 1
            }
        }
        let score:Float = Float(count) / Float(labels.shape[0])
        return score
    }
}