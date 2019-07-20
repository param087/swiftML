import TensorFlow

/// Principal Component Analysis
public class PCA {
  
    public var componentCount: Int
    public var whiten: Bool

    public var sampleCount: Int
    public var featureCount: Int
    public var mean: Tensor<Float>
    public var noiseVariance: Tensor<Float>
    public var components: Tensor<Float>
    public var expainedVariance: Tensor<Float>
    public var expainedVarianceRatio: Tensor<Float>
    public var singularValues: Tensor<Float>  
    
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
    public init(componentCount: Int = 0, whiten: Bool = false) {

        self.componentCount = componentCount
        self.whiten = whiten
        
        self.mean = Tensor<Float>([0])
        self.noiseVariance = Tensor<Float>([0])
        self.expainedVariance = Tensor<Float>([0])
        self.expainedVarianceRatio = Tensor<Float>([0])
        self.singularValues = Tensor<Float>([0])
        self.components = Tensor<Float>([0])
        self.sampleCount = 0
        self.featureCount = 0
    }
  
    /// Fit Principal Component Analysis.
    ///
    /// - Parameter data: Training Tensor<Float> of shape[sampleCount, featureCount],
    ///   where sampleCount is the number of sample and featureCount is the number of
    ///   features.
    public func fit(data: Tensor<Float>) {
        
        let sampleCount = data.shape.dimensions[0]
        let featureCount = data.shape.dimensions[1]
        let componentCount = self.componentCount
        var s: Tensor<Float>
        var u: Tensor<Float>
        var v: Tensor<Float>
        
        precondition(componentCount <= featureCount,
            "Number of Components must be greater than Number of features")
        
        self.mean = data.mean(alongAxes: 0)
        
        let tempData = data - self.mean
        let svd = Raw.svd(tempData)
        s = svd.s
        u = svd.u
        v = svd.v.transposed()
        
        (u, v) = svdFlip(u: u, v: v)
        
        let components = v
        let expainedVariance = pow(s, 2) / Tensor<Float>(Float(sampleCount - 1))
        let totalVariance = expainedVariance.sum()
        let expainedVarianceRatio = expainedVariance / totalVariance
        let singularValues = s
        
        if componentCount < min(featureCount, sampleCount) {
            self.noiseVariance = expainedVariance.flattened()
                .slice(lowerBounds: [componentCount], upperBounds: [s.shape[0]]).mean()
        } else {
            self.noiseVariance = Tensor<Float>(0.0)
        }
        
        self.sampleCount = sampleCount
        self.featureCount = featureCount
        self.components = components
            .slice(lowerBounds: [0, 0], upperBounds: [componentCount, components.shape[1]])
        self.componentCount = componentCount
        self.expainedVariance = expainedVariance.flattened()
            .slice(lowerBounds: [0], upperBounds: [self.componentCount])
        self.expainedVarianceRatio = expainedVarianceRatio.flattened()
            .slice(lowerBounds: [0], upperBounds: [self.componentCount])
        self.singularValues = singularValues.flattened()
            .slice(lowerBounds: [0], upperBounds: [self.componentCount])
    }

    /// Returns dimensionally reduce data
    ///
    /// - Parameter data: Input Tensor<Float> of shape[sampleCount, featureCount], where
    ///   sampleCount is the number of sample and featureCount is the number of features.
    /// - Returns: Dimensionally reduced data.
    public func transformation(for data: Tensor<Float>) -> Tensor<Float> {

        var transformedData = matmul((data - self.mean), self.components.transposed())

        if self.whiten {
            transformedData = transformedData / sqrt(self.expainedVariance)
        }
        
        return transformedData
    }

    /// Return transform data to its original space.
    ///
    /// - Parameter data: Input Tensor<Float> of shape[sampleCount, featureCount], where
    ///   sampleCount is the number of sample and featureCount is the number of features.
    /// - Returns: Original data whose transform would be data.
    public func inverseTransform(data: Tensor<Float>) -> Tensor<Float> {
        
        if self.whiten {
            return matmul(data, sqrt(self.expainedVariance
                .reshaped(to: [self.expainedVariance.shape[1], 1]) * self.components)) + self.mean
        } else {
            return matmul(data, self.components) + self.mean
        }
    }
}