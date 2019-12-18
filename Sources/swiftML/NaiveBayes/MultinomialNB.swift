import TensorFlow

/// Multinomial naive bayes classifier.
///
/// Multinomial naive bayes classifier used to classify discrete features.
///
/// Reference: ["Multinomial Naive bayes"](
/// https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)
public class MultinomialNB {
    /// Additive smoothing parameter.
    public var alpha: Float
    /// The prior log probability for each class.
    public var classLogPrior: Tensor<Float>
    /// Log probability of each word.
    public var featureLogProb: Tensor<Float>
    /// Unique classes in target value set.
    public var classes: Tensor<Int32>
    /// Tensor contains the index of class in classes.
    public var indices: Tensor<Int32>
  
    /// Create a multinomial naive bayes model.
    ///
    /// - Parameter alpha: Additive smoothing parameter, default to `1.0`.
    public init(
        alpha:Float = 1.0
    ) {
        self.alpha = alpha
        self.classes = Tensor<Int32>([0])
        self.indices = Tensor<Int32>([0])
        self.classLogPrior = Tensor<Float>([0.0])
        self.featureLogProb = Tensor<Float>([0.0])
    }
  
    /// Fit a multinomial naive bayes classifier model.
    ///
    /// - Parameters:
    ///   - data: Training data with shape `[sample count, feature count]`.
    ///   - labels: Target values with shape `[sample count]`.
    public func fit(data: Tensor<Float>, labels: Tensor<Int32>) {
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have same sample count.")
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(data.shape[1] >= 1,
            "Data must have feature count greater than or equal to one.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")

        // find unique classes in target values.
        (self.classes, self.indices) = _Raw.unique(labels.flattened())
        
        precondition(self.classes.shape[0] > 1, "Labels must have more than one class.")

        // Initialize the classLogPrior and featureLogProb based on feature count and sample count.
        var separated = [[Tensor<Float>]]()
        self.classLogPrior = Tensor<Float>(zeros: [self.classes.shape[0]])
        self.featureLogProb = Tensor<Float>(zeros: [self.classes.shape[0], data.shape[1]])
      
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
      
        let sampleCount = data.shape[0]
      
        // Calculate `logp^(c)`, the prior log probability for each class.
        for index in 0..<separated.count {
            let temp = Float(separated[index].count) / Float(sampleCount)
            self.classLogPrior[index] = Tensor<Float>(log(temp))
        }
      
        var count = Tensor<Float>(zeros: [separated.count, data.shape[1]])
  
        // Count each word for each class and add `alpha` as smoothing.
        for i in 0..<separated.count {
            for j in 0..<data.shape[1] {
                var temp = Tensor<Float>(0.0)
                for k in 0..<separated[i].count {
                    temp = temp + separated[i][k][j]
                }
                count[i][j] = temp + self.alpha
            }
        }

        // Calculate the log probability of each word, `logp^(t|c)`.
        for i in 0..<count.shape[0] {
            var sum = Tensor<Float>(0.0)
            
            for j in 0..<count[i].shape[0] {
                sum = sum + count[i][j]
            }
            for j in 0..<count[i].shape[0] {
                self.featureLogProb[i][j] = log(count[i][j] / sum)
            }
        }
    }

    /// Returns log-probability estimates for the test data.
    ///
    /// - Parameter data: Test data with shape `[sample count, feature count]`.
    /// - Returns: log-probability estimates for the test data.
    public func predictLogProba(data: Tensor<Float>) -> Tensor<Float>{
        var predictLogProb = Tensor<Float>(zeros: [data.shape[0], self.classes.shape[0]])

        for i in 0..<data.shape[0] {
            let temp = self.featureLogProb * data[i]
            predictLogProb[i] = temp.sum(alongAxes: 1).flattened() + self.classLogPrior
        }

        return predictLogProb
    }

    /// Returns classified input data.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    /// - Returns: Classification of input data.
    public func prediction(for data: Tensor<Float>) -> Tensor<Int32> {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        
        let predictLogProb = self.predictLogProba(data: data)
        
        var prediction = Tensor<Int32>(zeros: [data.shape[0]])
        for i in 0..<data.shape[0] {
            prediction[i] = self.classes[Int(predictLogProb[i].argmax().scalarized())]
        }
        
        return prediction
    }

    /// Returns mean accuracy on the given test data and labels.
    ///
    /// - Parameters:
    ///   - data: Sample data with shape `[sample count, feature count]`.
    ///   - labels: Target values with shape `[sample count]`.
    /// - Returns: Returns the mean accuracy on the given test data and labels.
    public func score(data: Tensor<Float>, labels: Tensor<Int32>) -> Float {
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have the same sample count.")
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")

        let result = self.prediction(for: data)
        var count: Int = 0
        for i in 0..<result.shape[0] {
            if result[i] == labels[i] {
                count = count + 1
            }
        }
        let score: Float = Float(count) / Float(labels.shape[0])
        return score
    }
}