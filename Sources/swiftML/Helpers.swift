import TensorFlow

/// The generalized inverse of a matrix using its singular-value decomposition (SVD).
///
/// Return the (Moore-Penrose) pseudo-inverse of a matrix.
///
/// - Parameter x: The matrix to pseudo-inverse.
/// - Returns: Pseudo-inverse matrix.
public func matrixPseudoInverse(_ data: Tensor<Float>) -> Tensor<Float> {
    let svd = Raw.svd(data)
    let diag = Raw.diag(diagonal: svd.s)
    return matmul(matmul(svd.v, Raw.matrixInverse(diag)), svd.u.transposed())
}

/// Return Sign correction to ensure deterministic output from SVD.
///
/// - Parameters
///   - u: The tensor containing of left singular vectors for each matrix.
///   - v: The tensor containing of right singular vectors for each matrix.
/// - Returns: Sign correction to ensure deterministic output.
public func svdFlip(
    u: Tensor<Double>,
    v: Tensor<Double>,
    uBasedDecision: Bool = true
) -> (Tensor<Double>, Tensor<Double>) {
  
    var U = u
    var V = v
    
    if uBasedDecision {
    
        let maxAbsCols = abs(u).argmax(squeezingAxis: 0)
        var colValueForSign = Tensor<Double>(zeros: [u.shape[1]])
        
        for i in 0..<u.shape[1] {
            colValueForSign[i] = u[Int(maxAbsCols[i].scalarized()), i]
        }
        
        let signs = Raw.sign(colValueForSign)
        U = U * signs
        V = V * signs.reshaped(to: [u.shape[1], 1])
        
    } else {
        
        let maxAbsRows = abs(v).argmax(squeezingAxis: 1)
        var rowValueForSign = Tensor<Double>(zeros: [u.shape[0]])
        
        for i in 0..<u.shape[0] {
            rowValueForSign[i] = v[i, Int(maxAbsRows[i].scalarized())]
        }
        
        let signs = Raw.sign(rowValueForSign)
        U = U * signs
        V = V * signs.reshaped(to: [u.shape[1], 1])
    }
    
    return (U, V)
}
  
/// Return the minkowski distance based on value of p.
///
/// - Parameters
///   - a: Input tensor to find distance.
///   - b: Input tensor to find distance.
///   - p: The distance metric to use for the tree, default set to 2.
/// - Returns: Minkowski distance based on value of p.
public func minkowskiDistance(_ a: Tensor<Float>, _ b: Tensor<Float>, _ p: Int ) -> Tensor<Float> {
    return pow(pow(abs(b - a), Float(p)).sum(), 1.0/Float(p))
}

/// Return the euclidean distance.
///
/// - Parameters
///   - a: Input tensor to find distance.
///   - b: Input tensor to find distance.
/// - Returns: euclidean distance.
public func euclideanDistance(_ a: Tensor<Float>, _ b: Tensor<Float>) -> Tensor<Float> {
    return minkowskiDistance(a, b, 2)
}