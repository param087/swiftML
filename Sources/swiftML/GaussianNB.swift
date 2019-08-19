import TensorFlow

/// Gaussian naive bayes classifier.
///
/// Gaussian naive bayes classifier used to classify continuous features.
public class GaussianNB {
    /// Unique classes in target value set.
    public var classes: Tensor<Int32>
    /// Tensor contains the index of class in classes.
    public var indices: Tensor<Int32>
    /// The mean and the standard deviation of each feature of each class.
    public var model: Tensor<Float>
  
    /// Create a Gaussian naive bayes model.
    public init() {
        self.classes = Tensor<Int32>([0])
        self.indices = Tensor<Int32>([0])
        self.model = Tensor<Float>([0.0])
    }

    /// Fit a Gaussian naive bayes classifier model.
    ///
    /// - Parameters
    ///   - data: Training data with shape `[sample count, feature count]`.
    ///   - labels: Target value with shape `[sample count]`.
    public func fit(data: Tensor<Float>, labels: Tensor<Int32>) {
        precondition(data.shape[0] == labels.shape[0], 
            "Data and labels must have same sample count.")
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(data.shape[1] >= 1,
            "Data must have feature count greater than or equal to one.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")

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

    /// Returns a log of gaussian distribution.
    ///
    /// - Parameters
    ///   - data: Input data to find gausssian distribution.
    ///   - mean: Mean of input tensor.
    ///   - std: Standard deviation of input data.
    /// - Returns: Log of gaussian distribution.
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
    /// - Parameter data: Input data to predict log probability.
    /// - Return: predicted log probability.
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
  
    /// Returns classification.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    /// - Returns: prediction of input data.
    public func prediction(for data: Tensor<Float>) -> Tensor<Int32> {
        precondition(data.shape[0] > 0, "Data must be non-empty.")

        var labels = Tensor<Int32>(zeros: [data.shape[0]])
        let predLogProb = self.predictLogProba(data: data)
        
        for i in 0..<data.shape[0] {
            labels[i] = self.classes[Int(predLogProb[i].argmax().scalarized())]
        }
        return labels
    }

    /// Returns mean accuracy on the given input data and labels.
    ///
    /// - Parameters
    ///   - data: Sample data with shape `[sample count, feature count]`.
    ///   - labels: Target label with shape `[sample count]`.
    /// - Returns: Returns the mean accuracy on the given input data and labels.
    public func score(data: Tensor<Float>, labels: Tensor<Int32>) -> Float {
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have same sample count.")
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")

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