import TensorFlow

/// Principal Component Analysis
///
/// Reference: ["Principal Component Analysis"](
/// https://en.wikipedia.org/wiki/Principal_component_analysis)
public class PCA {
    /// The estimated number of components.
    public var componentCount: Int
    /// Whitening will remove some information from the transformed signal (the relative variance
    /// scales of the components) but can sometime improve the predictive accuracy of the
    /// downstream estimators by making their data respect some hard-wired assumptions.
    public var whiten: Bool
    /// Number of samples in the training data.
    public var sampleCount: Int
    /// Number of feature in the training data.
    public var featureCount: Int
    /// Per-feature empirical mean, estimated from the training set.
    public var mean: Tensor<Double>
    /// The estimated noise covariance.
    public var noiseVariance: Tensor<Double>
    /// Principal axes in feature space, representing the directions of maximum variance in the
    /// data.
    public var components: Tensor<Double>
    /// The amount of variance explained by each of the selected components.
    public var explainedVariance: Tensor<Double>
    /// Percentage of variance explained by each of the selected components.
    public var explainedVarianceRatio: Tensor<Double>
    /// The singular values corresponding to each of the selected components.
    public var singularValues: Tensor<Double>  
    
    /// Create Principal Component Analysis model.
    ///
    /// - Parameters:
    ///   - componentCount: Number of components to keep.
    ///   - whiten: When `true` (`false` by default) the `components` vectors are multiplied by the
    ///     square root of sample count and then divided by the singular values to ensure 
    ///     uncorrelated outputs with unit component-wise variances. Whitening will remove some
    ///     information from the transformed signal (the relative variance scales of the
    ///     components) but can sometime improve the predictive accuracy of the downstream
    ///     estimators by making their data respect some hard-wired assumptions.
    public init(
        componentCount: Int = 0,
        whiten: Bool = false
    ) {
        self.componentCount = componentCount
        self.whiten = whiten
        
        self.mean = Tensor<Double>([0])
        self.noiseVariance = Tensor<Double>([0])
        self.explainedVariance = Tensor<Double>([0])
        self.explainedVarianceRatio = Tensor<Double>([0])
        self.singularValues = Tensor<Double>([0])
        self.components = Tensor<Double>([0])
        self.sampleCount = 0
        self.featureCount = 0
    }
    
    /// Returns the log-likelihood of a rank over given dataset and spectrum.
    ///
    /// - Parameters:
    ///   - spectrum: The amount of variance explained by each of the seleted components.
    ///   - rank: Test rank value.
    ///   - sampleCount: The sample count.
    ///   - featureCount: The features count.
    /// - Returns: Log-likelihood of rank over given dataset.
    func assessDimension(
        _ spectrum: Tensor<Double>,
        _ rank: Int,
        _ sampleCount: Int,
        _ featureCount: Int
    ) -> Tensor<Double> {
        var pu = -Double(rank) * Double(log(2.0))
        
        for i in 0..<rank {
            let logGamma = Raw.lgamma(Tensor<Double>(Double(featureCount - i) / 2.0)).scalarized()
            pu = pu + (logGamma - Double(log(Float.pi)) * Double(featureCount - i) / 2.0)
        }
      
        var pl = Double(log(spectrum.slice(lowerBounds: [0], upperBounds: [rank])).sum())
        pl = -1 * pl! * Double(sampleCount) / 2.0
      
        var pv: Double = 0.0
        var v: Double = 0.0
      
        if rank == featureCount {
            pv = 0.0
            v = 1.0
        } else {
            v = spectrum.slice(lowerBounds: [rank], upperBounds: [spectrum.shape[0]])
                    .sum().scalarized() / Double(featureCount - rank)
            pv = -1.0 * Double(log(v)) * Double(sampleCount) * Double(featureCount - rank) / 2.0
        }
      
        let m = Double(featureCount) * Double(rank) - Double(rank) * (Double(rank) + 1.0) / 2.0
        let pp = Double(log(2.0 * Float.pi)) * (m + Double(rank) + 1.0) / 2.0
      
        var pa: Double = 0.0
        var spectrumCopy = spectrum
      
        for i in rank..<spectrum.shape[0] {
            spectrumCopy[i] = Tensor<Double>(v)
        }
      
        for i in 0..<rank {
            for j in (i+1)..<spectrum.shape[0] {
                let spectrumDiff = (spectrum[i] - spectrum[j]).scalarized()
                let inverseSpectrumCopyDiff =  (1.0 / spectrumCopy[j] - 1.0 
                    / spectrumCopy[i]).scalarized()
                pa = pa + Double(log( spectrumDiff * inverseSpectrumCopyDiff))
                    + Double(log(Float(sampleCount)))
              
            }
        }
      
        let temp = Double(rank) * Double(log(Float(sampleCount))) / 2.0
        let logLikelyhood = pu + pl! + pv + pp - pa / 2.0 - temp
        
        return Tensor<Double>(logLikelyhood)       
    }
  
    /// Returns the number of components best describe the dataset.
    ///
    /// Reference: ["Automatic Choice of Dimensionality for PCA"](
    /// https://pdfs.semanticscholar.org/cbaa/eb023b8a07ee05a617791f7740a176a1de1b.pdf)
    ///
    /// - Parameters:
    ///   - spectrum: The amount of variance explained by each of the seleted components.
    ///   - sampleCount: The sample count.
    ///   - featureCount: The feature count.
    /// - Returns: The number of components best describe the dataset.
    func inferDimension(
        spectrum: Tensor<Double>,
        sampleCount: Int,
        featureCount: Int
    ) -> Int {
        let spectrumCount = spectrum.shape[0]
        var logLikelihood = Tensor<Double>(zeros: [spectrumCount])
        
        for i in 0..<spectrumCount {
            logLikelihood[i] = assessDimension(spectrum, i, sampleCount, featureCount)
        }
        
        return Int(logLikelihood.argmax().scalarized())
    }
  
    /// Fit a Principal Component Analysis.
    ///
    /// - Parameter data: Training data with shape `[sample count, feature count]`.
    public func fit(data: Tensor<Double>) {
        self.sampleCount = data.shape.dimensions[0]
        self.featureCount = data.shape.dimensions[1]
        /// Local component count and can be modified based on type of componet selection algorithm
        /// execuation.
        var componentCount = self.componentCount
        /// Singular values.
        var s: Tensor<Double>
        /// Left singular vectors.
        var u: Tensor<Double>
        /// Right singular vectors.
        var v: Tensor<Double>
        
        precondition(componentCount >= 0, "Component count must be non-negative.")
        precondition(componentCount <= featureCount,
            "Component count must be smaller than feature counts.")
        
        self.mean = data.mean(alongAxes: 0)
        
        let dataDeviation = data - self.mean
        (s, u, v) = deterministicSvd(dataDeviation)
        
        let components = v
        let explainedVariance = pow(s, 2) / Tensor<Double>(Double(sampleCount - 1))
        let totalVariance = explainedVariance.sum()
        let explainedVarianceRatio = explainedVariance / totalVariance
        let singularValues = s
 
        if componentCount == 0 {
            componentCount = self.inferDimension(spectrum: explainedVariance,
                sampleCount: sampleCount, featureCount: featureCount)
        }
        
        if componentCount < min(featureCount, sampleCount) {
            self.noiseVariance = explainedVariance.flattened()
                .slice(lowerBounds: [componentCount], upperBounds: [s.shape[0]]).mean()
        } else {
            self.noiseVariance = Tensor<Double>(0.0)
        }
        
        self.components = components
            .slice(lowerBounds: [0, 0], upperBounds: [componentCount, components.shape[1]])
        self.componentCount = componentCount
        self.explainedVariance = explainedVariance.flattened()
            .slice(lowerBounds: [0], upperBounds: [self.componentCount])
        self.explainedVarianceRatio = explainedVarianceRatio.flattened()
            .slice(lowerBounds: [0], upperBounds: [self.componentCount])
        self.singularValues = singularValues.flattened()
            .slice(lowerBounds: [0], upperBounds: [self.componentCount])
    }

    /// Returns dimensionally reduced data.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    /// - Returns: Dimensionally reduced data.
    public func transformation(for data: Tensor<Double>) -> Tensor<Double> {
        var transformedData = matmul((data - self.mean), self.components.transposed())

        if self.whiten {
            transformedData = transformedData / sqrt(self.explainedVariance)
        }
        
        return transformedData
    }

    /// Returns transform data to its original space.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    /// - Returns: Original data whose transform would be data.
    public func inverseTransformation(for data: Tensor<Double>) -> Tensor<Double> {
        if self.whiten {
            return matmul(data, sqrt(self.explainedVariance
                .reshaped(to: [self.explainedVariance.shape[1], 1]) * self.components)) + self.mean
        } else {
            return matmul(data, self.components) + self.mean
        }
    }
}