import TensorFlow

/// A Logistic regression classifier.
///
/// In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme.
public class LogisticRegression {

    var learningRate: Float
    var iterations: Int
    var fitIntercept: Bool
    // The tensor of unique classes
    var classes: Tensor<Int32>
    // The tensor contains the index to unique classes
    var indices: Tensor<Int32>
    public var weights = [Tensor<Float>]()
    
    /// Creates a logistic regression model.
    ///
    /// - Parameters
    ///   - iterations: The number of iterations for gradient descent, default to 0.1.
    ///   - learningRate: The learning rate for gardient descent, default to 1000.
    ///   - fitIntercept: whether to calculate the intercept for this model. If set to False, no
    ///     intercept will be used in calculations, default set to true.
    public init(
        learningRate: Float = 0.1,
        iterations: Int = 1000,
        fitIntercept: Bool = true
    ) {

        precondition(iterations > 0, "Number of iterations must be greater than zero.")
        precondition(learningRate >= 0, "Learning Rate must be non-negative.")

        self.learningRate = learningRate
        self.iterations = iterations
        self.fitIntercept = fitIntercept
        self.classes = Tensor<Int32>(0)
        self.indices = Tensor<Int32>(0)
    }
    
    /// Fit logistic regression model.
    ///
    /// - Parameters
    ///   - data: Training data Tensor<Flaot> of shape [number of samples, number of features].
    ///   - labels: Target value Tensor<Flaot> of shape [number of samples, 1].
    public func fit(data: Tensor<Float>, labels: Tensor<Float>) {

        precondition(data.shape[0] == labels.shape[0],
            "data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "data must be non-empty.")
        precondition(data.shape[1] >= 1, "data must have atleast single feature.")
        precondition(labels.shape[0] > 0, "labels must be non-empty.")
        precondition(labels.shape[1] == 1, "labels must have single target.")

        var modifiedData = data

        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            modifiedData = ones.concatenated(with: data, alongAxis: -1)
        }
        let numberOfSamples = Float(modifiedData.shape[0])

        let labels = Tensor<Int32>(labels)
        (self.classes, self.indices) = Raw.unique(labels.flattened())

        precondition(self.classes.shape[0] >= 2, "labels must contains atleast 2 classes")
        
        // loop through each class, uses the one-vs-rest (OvR) scheme. 
        for i in 0..<self.classes.shape[0] {

            let condition = Raw.equal(labels, classes[i])
            let t = Tensor<Int32>(ones:[labels.shape[0],1])
            let e = Tensor<Int32>(zeros:[labels.shape[0],1])

            // create temparary label for one-vs-rest scheme, based on class the selected class
            // labeled as one while rest as zeros.
            var tempLabels = Tensor<Float>(Raw.select(condition: condition, t: t, e: e))
            tempLabels = tempLabels.reshaped(to: [tempLabels.shape[0], 1])
    
            // weights of selected class in one-vs-rests scheme.
            var tempWeights = Tensor<Float>(ones: [modifiedData.shape[1], 1])

            for _ in 0..<self.iterations {
                let output = matmul(modifiedData, tempWeights)
                let errors = tempLabels - sigmoid(output)
                tempWeights = tempWeights +
                    ((self.learningRate / numberOfSamples) *
                        matmul(modifiedData.transposed(), errors))
            }

            self.weights.append(tempWeights)
        }
    }
    
    /// Return the prediction of single sample
    ///
    /// - Parameters data: Sample tuple Tensor<Flaot> of shape [1, number of features].
    /// - Returns: Predicted class label.
    public func predictSingleSample(_ data: Tensor<Float>) -> Tensor<Float> {
        var output = Tensor<Float>(zeros:[weights.count, 1])
        var counter: Int = 0

        for weightIndex in 0..<self.weights.count {
            output[counter] = matmul(
                data.reshaped(to: [1, data.shape[0]]), self.weights[weightIndex])[0]
            counter = counter + 1
        }

        let index: Int = Int(output.argmax().scalarized())
        let classLabel: Tensor<Int32> = self.classes[index]

        return Tensor<Float>([Tensor<Float>(classLabel)])
    }
    
    /// Returns predict using logistic regression classifier.
    ///
    /// - Parameter data: Smaple data tensor of shape [number of samples, number of features].
    /// - Returns: Predicted target tensor of class labels.
    public func prediction(for data: Tensor<Float>) -> Tensor<Float> {

        precondition(data.shape[0] > 0, "data must be non-empty.")

        var modifiedData = data
        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            modifiedData = ones.concatenated(with: data, alongAxis: -1)
        }
        
        var result = Tensor<Float>(zeros: [modifiedData.shape[0], 1])
        for i in 0..<modifiedData.shape[0] {
            result[i] = self.predictSingleSample(modifiedData[i])
        }
        return result
    }

    /// Returns mean accuracy on the given test data and labels.
    ///
    /// - Parameters
    ///   - data: Sample tensor of shape [number of samples, number of features].
    ///   - labels: Target label tensor of shape [number of samples, 1].
    /// - Returns: Returns the mean accuracy on the given test data and labels.
    public func score(data: Tensor<Float>, labels: Tensor<Float>) -> Float {

        precondition(data.shape[0] == labels.shape[0],
            "data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "data must be non-empty.")
        precondition(labels.shape[0] > 0, "labels must be non-empty.")

        let result = self.prediction(for: data)
        var count: Int = 0
        for i in 0..<result.shape[0] {
            if result[i] == labels[i] {
                count = count + 1
            }
        }
        let score: Float = Float(count)/Float(labels.shape[0])
        return score
    }
}