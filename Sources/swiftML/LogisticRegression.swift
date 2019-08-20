import TensorFlow

/// A Logistic regression classifier.
///
/// In the multiclass classification, the training algorithm uses the one-vs-rest (OvR)
/// scheme.
public class LogisticRegression {
    ///  The linear rate for gradient descent.
    var learningRate: Float
    /// The number of iterations for gradient descent.
    var iterationCount: Int
    /// Whether to calculate the intercept for this model.
    var fitIntercept: Bool
    /// The tensor of unique classes.
    var classes: Tensor<Int32>
    /// The tensor contains the index to unique classes.
    var indices: Tensor<Int32>
    /// The weights array for the model contains weights of each class agains the rest.
    public var weights = [Tensor<Float>]()
    
    /// Creates a logistic regression model.
    ///
    /// - Parameters.
    ///   - iterationCount: The number of iterations for gradient descent, default to `1000`.
    ///   - learningRate: The learning rate for gardient descent, default to `0.1`
    ///   - fitIntercept: Whether to calculate the intercept for this model. If set to `false`, no
    ///     intercept will be used in calculations, default set to `true`.
    public init(
        iterationCount: Int = 1000,
        learningRate: Float = 0.1,
        fitIntercept: Bool = true
    ) {
        precondition(iterationCount > 0, "Iteration count must be positive.")
        precondition(learningRate >= 0, "Learning rate must be non-negative.")
        self.learningRate = learningRate
        self.iterationCount = iterationCount
        self.fitIntercept = fitIntercept
        self.classes = Tensor<Int32>(0)
        self.indices = Tensor<Int32>(0)
    }
    
    /// Fit a logistic regression model.
    ///
    /// - Parameters
    ///   - data: Training data with shape `[sample count, feature count]`.
    ///   - labels: Target value with shape `[sample count, 1]`.
    public func fit(data: Tensor<Float>, labels: Tensor<Int32>) {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(data.shape[1] >= 1,
            "Data must have feature count greater than or equal to one.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")
        precondition(labels.shape[1] == 1, "Labels must have single target feature.")
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have the same sample count.")

        var data = data

        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            data = ones.concatenated(with: data, alongAxis: -1)
        }
        let sampleCount = Float(data.shape[0])

        (self.classes, self.indices) = Raw.unique(labels.flattened())

        precondition(self.classes.shape[0] >= 2, "Labels must have atleast two classes")
        
        /// Loop through each class and apply one-vs-rest (OvR) scheme. 
        for i in 0..<self.classes.shape[0] {
            let condition = Raw.equal(labels, classes[i])
            let t = Tensor<Int32>(ones: [labels.shape[0], 1])
            let e = Tensor<Int32>(zeros: [labels.shape[0], 1])

            /// Create temparary labels for one-vs-rest scheme, based on class the selected class
            /// labeled as `1` while rest as `0`.
            var tempLabels = Tensor<Float>(Raw.select(condition: condition, t: t, e: e))
            tempLabels = tempLabels.reshaped(to: [tempLabels.shape[0], 1])
    
            /// weights of selected class in one-vs-rests scheme.
            var tempWeights = Tensor<Float>(ones: [data.shape[1], 1])

            for _ in 0..<self.iterationCount {
                let output = matmul(data, tempWeights)
                let errors = tempLabels - sigmoid(output)
                tempWeights = tempWeights +
                    ((self.learningRate / sampleCount) *
                        matmul(data.transposed(), errors))
            }

            self.weights.append(tempWeights)
        }
    }
    
    /// Return the prediction for a single sample.
    ///
    /// - Parameters data: Single sample data with shape `[1, feature count]`.
    /// - Returns: Predicted class label.
    public func predictSingleSample(_ data: Tensor<Float>) -> Tensor<Int32> {
        var output = Tensor<Float>(zeros: [weights.count, 1])
        var counter: Int = 0

        for weightIndex in 0..<self.weights.count {
            output[counter] = matmul(
                data.reshaped(to: [1, data.shape[0]]), self.weights[weightIndex])[0]
            counter = counter + 1
        }

        let index: Int = Int(output.argmax().scalarized())
        let classLabel: Tensor<Int32> = self.classes[index]

        return Tensor<Int32>([classLabel])
    }
    
    /// Returns prediction using logistic regression classifier.
    ///
    /// - Parameter data: Smaple data with shape `[sample count, feature count]`.
    /// - Returns: Predicted class label of target values.
    public func prediction(for data: Tensor<Float>) -> Tensor<Int32> {
        precondition(data.shape[0] > 0, "Data must be non-empty.")

        var data = data
        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            data = ones.concatenated(with: data, alongAxis: -1)
        }
        
        var result = Tensor<Int32>(zeros: [data.shape[0], 1])
        for i in 0..<data.shape[0] {
            result[i] = self.predictSingleSample(data[i])
        }
        return result
    }

    /// Returns mean accuracy on the given test data and labels.
    ///
    /// - Parameters
    ///   - data: Sample data with shape `[sample count, feature count].
    ///   - labels: Target label with shape `[sample count, 1]`.
    /// - Returns: Returns the mean accuracy on the given test data 1and labels.
    public func score(data: Tensor<Float>, labels: Tensor<Int32>) -> Float {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have the same sample count.")

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