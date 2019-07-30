import TensorFlow

/// The generalized inverse of a matrix using its singular-value decomposition (SVD).
///
/// Returns the (Moore-Penrose) pseudo-inverse of a matrix.
///
/// - Parameter data: The input matrix to be pseudo-inverse.
/// - Returns: Pseudo-inverse matrix.
public func matrixPseudoInverse(_ data: Tensor<Float>) -> Tensor<Float> {
    let svd = Raw.svd(data)
    let diag = Raw.diag(diagonal: svd.s)
    return matmul(matmul(svd.v, Raw.matrixInverse(diag)), svd.u.transposed())
}

/// Returns Sign correction to ensure deterministic output from SVD.
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

  
/// Returns the Minkowski distance between two tensors for the given distance metric `p`.
///
/// - Parameters
///   - a: The first tensor.
///   - b: The second tensor.
///   - p: The order of the norm of the difference: `||a - b||_p`.
/// - Returns: Minkowski distance based on value of `p`.
public func minkowskiDistance(_ a: Tensor<Float>, _ b: Tensor<Float>, p: Int ) -> Tensor<Float> {
    
    precondition(a.shape == b.shape, "Shape of both the inputs must be same.")
    precondition(p > 0, "p must be greater than zero.")

    return pow(pow(abs(b - a), Float(p)).sum(), 1.0/Float(p))
}

/// Returns the euclidean distance between two tensors.
///
/// - Parameters
///   - a: The first tensor.
///   - b: The second tensor.
/// - Returns: Euclidean distance: `||a - b||_2`.
public func euclideanDistance(_ a: Tensor<Float>, _ b: Tensor<Float>) -> Tensor<Float> {

    precondition(a.shape == b.shape, "Shape of both the inputs must be same.")
    
    return minkowskiDistance(a, b, p: 2)
}