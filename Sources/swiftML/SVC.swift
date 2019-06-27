import TensorFlow

public class SVC {
  
    public var X: Tensor<Float>
    public var Y: Tensor<Float>
    public var m: Int
    public var alphas: [Float]
    public var w: Tensor<Float>
    public var b: Float
    public var maxIter: Int
    public var epsilon: Float
    public var C : Float
    public var kernel: (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
    public var kernelType: String

  
    public init ( features: Tensor<Float>, 
                 labels: Tensor<Float>,
                 maxIter: Int,
                 epsilon: Float,
                 C: Float,
                 kernelType: String,
                 kernel: @escaping (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
                 ){
      
        self.X = features   //data set
        self.Y = labels //label set
        self.m = features.shape[0] //data set size
        self.alphas = Array(repeating: 0.0, count: self.m) //lagrange multipliers
        self.b = 0.0
        self.maxIter = maxIter
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.w = Tensor<Float>(0.0)
        self.kernelType = kernelType
    }
  
    /*method that optimizes the SVC using the SMO algorithm  
     and returns an array of support vectors and the iteration
    */
    public func fit() -> ([[Float]], Int){
    
        //vars pertaining to the data 
        let x = self.X.array
        let y = self.Y.array
        let n = x.shape[0]
        let d = x.shape[1]
    
        var alphas = Array<Float>(repeating: 0.0, count: n)
    
        var support_vectors : [[Float]]
    
        let kernel = self.kernel
    
        //counter for iterations
        var count = 0
    
        while true {
        
            count = count + 1 

            if(count > maxIter) {
                
                support_vectors = getSPVectors(alphas: alphas)
                
                return (support_vectors, count)
            }
         
            let alphas_prev = alphas
        
            for j in stride(from: 0, through: n-1, by: 1) {
          
                let i = getRandomInt(a: 0, b: n-1, z: j)
          
                let x_i = x[i].scalars
                let x_j = x[j].scalars
                let y_i = y[i].scalars[0]
                let y_j = y[j].scalars[0]
          
                let Tx_i = Tensor<Float>(x_i)
                let Tx_j = Tensor<Float>(x_j)
              
                let Tk_ij = kernel(Tx_i, Tx_i) + kernel(Tx_j, Tx_j) * 2 *
                            kernel(Tx_i, Tx_j)
          
                let k_ij = Tk_ij.array.scalars
          
                if k_ij[0] == 0 {
                    continue
                }
          
                let a_prime_i = alphas[i]
                let a_prime_j = alphas[j]
          
                let bounds = computeLH(C : self.C, 
                                       a_prime_i: a_prime_i, 
                                       a_prime_j: a_prime_j, 
                                       y_i: y_i, 
                                       y_j: y_j)
          
                let L = bounds.0
                let H = bounds.1
          
                self.w = computeW(alphas: Tensor<Float>(alphas), 
                                  x:self.X, 
                                  y:self.Y)
          
                self.b = computeB(x:self.X, y:self.Y, w: self.w)
          
                let E_i = computeE(x_k : x_i, y_k : y_i, w : self.w, b: self.b)
                let E_j = computeE(x_k : x_j, y_k : y_j, w : self.w, b: self.b)
          
          
                alphas[j] = a_prime_j + Float(y_j * (E_i - E_j))/k_ij[0]
                alphas[j] = max(alphas[j], L)
                alphas[j] = min(alphas[j], H)
          
                alphas[i] = a_prime_i + y_i*y_j * (a_prime_j - alphas[j])
          
                let d = Tensor<Float>(alphas) - Tensor<Float>(alphas_prev)
          
                let diff = sqrt(Raw.l2Loss(t: d)/2)*2
          
                if diff < self.epsilon {
                   break
                }
          
                if count >= self.maxIter {
                    print("Iteration number exceeded the max of iterations")
                    support_vectors = getSPVectors(alphas: alphas)
          
                    return (support_vectors, count) 
                }
          
                self.b = computeB(x:self.X, y:self.Y, w: self.w)
          
                if self.kernelType == "linear" || self.kernelType == "rbf"{
                    self.w = computeW(alphas: Tensor<Float>(alphas), 
                                      x:self.X, 
                                      y:self.Y)
                }
          
                support_vectors = getSPVectors(alphas: alphas)
          
                return (support_vectors, count)
         
          }
        
        
      }

  }
  
 
    //predicts class of an example   
    public func predictE(example : [Float]) -> Int {
         
        return computeH(X : example, w : self.w, b : self.b)
    
    }
  
    /// Returns classified test tensor.
    ///
    /// - Parameter X: Test Tensor<Float> of shape [number of samples, number of features].
    /// - Returns: classified class tensor.  
    public func predict(X: Tensor<Float>) -> Tensor<Float> {

        precondition(X.shape[0] > 0, "X must be non-empty.")

        var predictions : [Float] = []
        
        for i in 0..<X.shape[0] {
            let p = (Float)(predictE(example: X[i].scalars))
            predictions.append(p)
        }
        
        return Tensor<Float>(predictions)
    }
  
    //returns support vectors

    public func getSPVectors(alphas : [Float]) -> [[Float]] {
    
        var index = Array<Array<Float>>()
    
        for i in stride(from: 0, through: alphas.count-1, by: 1) {
            if alphas[i] > 0 {
                index.append(X[i].scalars)
            }
        }
   
        return index
    }
  
  
    public func computeE(x_k : [Float], 
                         y_k : Float, 
                         w : Tensor<Float>, 
                         b : Float) -> Float {
    
        let classify = Float(computeH(X:x_k, w:w, b:b))
    
        return classify - y_k
    }
  
  
    public func computeH(X : [Float], w : Tensor<Float>, b : Float) -> Int{
    
        let x = Tensor<Float>(X).reshaped(to: [1, self.X.shape[1]])
        let a = matmul(x, w.transposed())+b
    
        let h = a.scalars[0]
    
        if(h > 0) {
            return 1
        } else if (h == 0){
            return 0
        } else {
            return -1
        }

    }
  
    //computes weights
    public func computeW(alphas: Tensor<Float>, 
                         x: Tensor<Float>, 
                         y: Tensor<Float>) -> Tensor<Float>{

        let a = alphas.reshaped(to: [1, x.shape[0]])*y.reshaped(to: [1, y.shape[0]])
        return matmul(x.transposed(), 
                      a.reshaped(to: [x.shape[0], 1])).reshaped(to: [1,  x.shape[1]])
    }
  
    //computes bias
    public func computeB(x: Tensor<Float>, 
                         y: Tensor<Float>,  
                         w: Tensor<Float>) -> Float{
    
        let a = matmul(w.transposed().reshaped(to: [1,x.shape[1]]), 
                       x.transposed())
    
        let b = y - a
    
        return b.mean().scalars[0]
    
    }
  
    public func getRandomInt(a : Int, b : Int, z : Int) -> Int{
        var i = z
        var count=0
        while i == z && count<1000 {
            i = Int.random(in: a..<(b+1))
            count=count+1
        }
        return i
    }
  
    //computes bound for the alphas
    public func computeLH(C : Float, 
                          a_prime_i: Float, 
                          a_prime_j: Float, 
                          y_i: Float, 
                          y_j: Float) -> (Float, Float){
    
        if y_i != y_j {
            return (max(0, a_prime_j - a_prime_i), 
                    min(C, C - a_prime_i + a_prime_j))
        } else {
            return (max(0, a_prime_i + a_prime_j - C), 
                    min(C, a_prime_i + a_prime_j))
        } 
    
    }
  
  
    /*two types of kernels defined: further kernels can be defined in the future
      or provided directly by the user
    */
    public func linearKernel(x1 : Tensor<Float>, 
                             x2 : Tensor<Float>) -> Tensor<Float>{
        let d = x1.scalarCount
        return matmul(x1.reshaped(to: [1, d]), x2.reshaped(to: [d, 1]))
    }

  
    public func quadraticKernel(x1 : Tensor<Float>,
                                x2 : Tensor<Float>) -> Tensor<Float>{
        return matmul(x1, x2.transposed()).squared()
    }
  
    public func rbfGaussian(x1 : Tensor<Float>, 
                        x2 : Tensor<Float>
                        ) -> Tensor<Float>{
    
        let a = x1 - x2
  
        let n = a.shape[0]
        let m = a.shape[1]

        let b = (Float)(-1.0)*3*matmul(a.transposed(), a)
         
        return Raw.exp(b)
    
     }
  
    /// Returns Predict class labels for input samples.
    ///
    /// - Parameters
    ///   - X: Sample tensor of shape [number of samples, number of features].
    ///   - y: Target label tensor of shape [number of samples].
    /// - Returns: Returns the mean accuracy on the given test data and labels.
    public func score(X: Tensor<Float>, y: Tensor<Float>) -> Float {
        
        precondition(X.shape[0] == y.shape[0], "X and y must have same number of samples.")
        precondition(X.shape[0] > 0, "X must be non-empty.")
        precondition(y.shape[0] > 0, "y must be non-empty.")

        let yPred = self.predict(X: X)
        var count:Int = 0
        for i in 0..<X.shape[0] {
            if yPred[i] == y[i] {
                count = count + 1
            }
        }
        let score = Float(count) / Float(y.shape[0])
        return score
    }
  
}
