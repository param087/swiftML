import TensorFlow

/// Principal Component Analysis
public class PCA {
  
    public var nComponents: Int
    public var whiten: Bool

    public var nSamples: Int
    public var nFeatures: Int
    public var mean: Tensor<Float>
    public var noiseVariance: Tensor<Float>
    public var components: Tensor<Float>
    public var expainedVariance: Tensor<Float>
    public var expainedVarianceRatio: Tensor<Float>
    public var singularValues: Tensor<Float>  
    
    /// Create Principal Component Analysis model.
    ///
    /// - Parameters:
    ///   - nComponents:  Number of components to keep.
    ///   - whiten: When True (False by default) the `components` vectors are multiplied by the
    ///     square root of n_samples and then divided by the singular values to ensure uncorrelated
    ///     outputs with unit component-wise variances. Whitening will remove some information from
    ///     the transformed signal (the relative variance scales of the components) but can
    ///     sometime improve the predictive accuracy of the downstream estimators by making
    ///     their data respect some hard-wired assumptions.
    public init(nComponents: Int = 0, whiten: Bool = false) {

        self.nComponents = nComponents
        self.whiten = whiten
        
        self.mean = Tensor<Float>([0])
        self.noiseVariance = Tensor<Float>([0])
        self.expainedVariance = Tensor<Float>([0])
        self.expainedVarianceRatio = Tensor<Float>([0])
        self.singularValues = Tensor<Float>([0])
        self.components = Tensor<Float>([0])
        self.nSamples = 0
        self.nFeatures = 0
    }
  
    /// Fit Principal Component Analysis.
    ///
    /// - Parameter X: Training Tensor<Float> of shape[nSamples, nFeatures], where nSamples is the
    ///   number of sample and nFeatures is the number of features.
    public func fit(X: Tensor<Float>) {
        
        let nSamples = X.shape.dimensions[0]
        let nFeatures = X.shape.dimensions[1]
        let nComponents = self.nComponents
        var s: Tensor<Float>
        var u: Tensor<Float>
        var v: Tensor<Float>
        
        precondition(nComponents <= nFeatures,
            "Number of Components must be greater than Number of features")
        
        self.mean = X.mean(alongAxes: 0)
        
        let XTemp = X - self.mean
        let svd = Raw.svd(XTemp)
        s = svd.s
        u = svd.u
        v = svd.v.transposed()
        
        (u, v) = svdFlip(u: u, v: v)
        
        let components = v
        let expainedVariance = pow(s, 2) / Tensor<Float>(Float(nSamples - 1))
        let totalVar = expainedVariance.sum()
        let expainedVarianceRatio = expainedVariance / totalVar
        let singularValues = s
        
        if nComponents < min(nFeatures, nSamples) {
            self.noiseVariance = expainedVariance.flattened()
                .slice(lowerBounds: [nComponents], upperBounds: [s.shape[0]]).mean()
        } else {
            self.noiseVariance = Tensor<Float>(0.0)
        }
        
        self.nSamples = nSamples
        self.nFeatures = nFeatures
        self.components = components
            .slice(lowerBounds: [0, 0], upperBounds: [nComponents, components.shape[1]])
        self.nComponents = nComponents
        self.expainedVariance = expainedVariance.flattened()
            .slice(lowerBounds: [0], upperBounds: [self.nComponents])
        self.expainedVarianceRatio = expainedVarianceRatio.flattened()
            .slice(lowerBounds: [0], upperBounds: [self.nComponents])
        self.singularValues = singularValues.flattened()
            .slice(lowerBounds: [0], upperBounds: [self.nComponents])
    }

    /// Returns dimensionally reduce X
    ///
    /// - Parameter X: Input Tensor<Float> of shape[nSamples, nFeatures], where nSamples is the
    ///   number of sample and nFeatures is the number of features.
    /// - Returns: Dimensionally reduced X.
    public func transform(X: Tensor<Float>) -> Tensor<Float> {
        
        let XTemp = X - self.mean
        var X_transformed = matmul(XTemp, self.components.transposed())

        if self.whiten {
            X_transformed = X_transformed / sqrt(self.expainedVariance)
        }
        
        return X_transformed
    }

    /// Return transform data to its original space.
    ///
    /// - Parameter X: Input Tensor<Float> of shape[nSamples, nFeatures], where nSamples is the
    ///   number of sample and nFeatures is the number of features.
    /// - Returns: Input X_original whose transform would be X.
    public func inverseTransform(X: Tensor<Float>) -> Tensor<Float> {
        
        if self.whiten {
            return matmul(X, sqrt(self.expainedVariance
                .reshaped(to: [self.expainedVariance.shape[1], 1]) * self.components)) + self.mean
        } else {
            return matmul(X, self.components) + self.mean
        }
    }
}