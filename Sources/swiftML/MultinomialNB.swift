import TensorFlow

/// Multinomial naive bayes classifier.
///
/// Multinomial naive bayes classifier used to classify discrete features.
public class MultinomialNB {

    public var alpha: Float
    // The prior log probability for each class.
    public var classLogPrior: Tensor<Float>
    // Log probability of each word.
    public var featureLogProb: Tensor<Float>
    // Unique classes in target value set.
    public var classes: Tensor<Int32>
    // Tensor contains the index of class in classes.
    public var indices: Tensor<Int32>
  
    /// Create a multinomial naive model.
    ///
    /// - Parameter alpha: Additive smoothing parameter, default to 1.0.
    public init(
        alpha:Float = 1.0
    ) {
        self.alpha = alpha
        self.classes = Tensor<Int32>([0])
        self.indices = Tensor<Int32>([0])
        self.classLogPrior = Tensor<Float>([0.0])
        self.featureLogProb = Tensor<Float>([0.0])
    }
  
    /// Fit multinomial naive bayes classifier model.
    ///
    /// - Parameters
    ///   - data: Training data Tensor<Float> of shape [number of samples, number of features].
    ///   - labels: Target value Tensor<Float> of shape [number of samples].
    public func fit(data: Tensor<Float>, labels: Tensor<Float>) {
      
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "Data must be non-empty.")
        precondition(data.shape[1] >= 1, "Data must have atleast single feature.")
        precondition(labels.shape[0] > 0, "Labels must be non-empty.")

        let labels = Tensor<Int32>(labels)
        // find unique classes in target values.
        (self.classes, self.indices) = Raw.unique(labels.flattened())
        
        precondition(self.classes.shape[0] > 1, "labels must have more than one classes.")

        // Initialize the classLogPrior and featureLogProb based on number of features and sample.
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
      
        // Calculate logp^(c) , the prior log probability for each class.
        for index in 0..<separated.count {
            let temp = Float(separated[index].count) / Float(sampleCount)
            self.classLogPrior[index] = Tensor<Float>(log(temp))
        }
      
        var count = Tensor<Float>(zeros: [separated.count, data.shape[1]])
  
        // Count each word for each class and add alpha as smoothing.
        for i in 0..<separated.count {
            for j in 0..<data.shape[1] {
                var temp = Tensor<Float>(0.0)
                for k in 0..<separated[i].count {
                    temp = temp + separated[i][k][j]
                }
                count[i][j] = temp + self.alpha
            }
        }

        // Calculate the log probability of each word, logp^(t|c)
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

    /// Returns log-probability estimates for the test tensor.
    ///
    /// - Parameter data: Text Tensor<Float> of shape [number of samples, number of features].
    public func predictLogProba(data: Tensor<Float>) -> Tensor<Float>{

        var predictLogProb = Tensor<Float>(zeros: [data.shape[0], self.classes.shape[0]])

        for i in 0..<data.shape[0] {
            let temp = self.featureLogProb * data[i]
            predictLogProb[i] = temp.sum(alongAxes: 1).flattened() + self.classLogPrior
        }

        return predictLogProb
    }

    /// Returns classified test tensor.
    ///
    /// - Parameter data: Test Tensor<Float> of shape [number of samples, number of features].
    /// - Returns: classified tensor.
    public func prediction(for data: Tensor<Float>) -> Tensor<Int32> {

        precondition(data.shape[0] > 0, "Data must be non-empty.")
        
        let predictLogProb = self.predictLogProba(data: data)
        
        var prediction = Tensor<Int32>(zeros: [data.shape[0]])
        for i in 0..<data.shape[0] {
            prediction[i] = self.classes[Int(predictLogProb[i].argmax().scalarized())]
        }
        
        return prediction
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
        precondition(data.shape[0] > 0, "Data must be non-empty.")
        precondition(labels.shape[0] > 0, "labels must be non-empty.")

        let result = Tensor<Float>(self.prediction(for: data))
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