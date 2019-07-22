import TensorFlow

/// Principal Component Analysis
public class PCA {
  
    public var componentCount: Int
    public var whiten: Bool

    public var sampleCount: Int
    public var featureCount: Int
    public var mean: Tensor<Double>
    public var noiseVariance: Tensor<Double>
    public var components: Tensor<Double>
    public var explainedVariance: Tensor<Double>
    public var explainedVarianceRatio: Tensor<Double>
    public var singularValues: Tensor<Double>  
    
    /// Create Principal Component Analysis model.
    ///
    /// - Parameters:
    ///   - componentCount: Number of components to keep.
    ///   - whiten: When True (False by default) the `components` vectors are multiplied by the
    ///     square root of sampleCount and then divided by the singular values to ensure 
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
    ///   - sampleCount: The number of samples.
    ///   - featureCount: The number of features.
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
  
    // This implements the method of `Thomas P. Minka:
    // Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
    /// Returns the number of components best describe the dataset.
    ///
    /// - Parameters:
    ///   - spectrum: The amount of variance explained by each of the seleted components.
    ///   - sampleCount: The number of samples.
    ///   - featureCount: The number of features.
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
  
    /// Fit Principal Component Analysis.
    ///
    /// - Parameter data: Training Tensor<Double> of shape[sampleCount, featureCount],
    ///   where sampleCount is the number of sample and featureCount is the number of
    ///   features.
    public func fit(data: Tensor<Double>) {
        
        let sampleCount = data.shape.dimensions[0]
        let featureCount = data.shape.dimensions[1]
        var componentCount = self.componentCount
        var s: Tensor<Double>
        var u: Tensor<Double>
        var v: Tensor<Double>
        
        precondition(componentCount <= featureCount,
            "Number of Components must be smaller than Number of features")
        
        self.mean = data.mean(alongAxes: 0)
        
        let tempData = data - self.mean
        let svd = Raw.svd(tempData)
        s = svd.s
        u = svd.u
        v = svd.v.transposed()
        (u, v) = svdFlip(u: u, v: v)
        
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
        
        self.sampleCount = sampleCount
        self.featureCount = featureCount
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

    /// Returns dimensionally reduce data
    ///
    /// - Parameter data: Input Tensor<Double> of shape[sampleCount, featureCount], where
    ///   sampleCount is the number of samples and featureCount is the number of features.
    /// - Returns: Dimensionally reduced data.
    public func transformation(for data: Tensor<Double>) -> Tensor<Double> {

        var transformedData = matmul((data - self.mean), self.components.transposed())

        if self.whiten {
            transformedData = transformedData / sqrt(self.explainedVariance)
        }
        
        return transformedData
    }

    /// Return transform data to its original space.
    ///
    /// - Parameter data: Input Tensor<Double> of shape[sampleCount, featureCount], where
    ///   sampleCount is the number of sample and featureCount is the number of features.
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