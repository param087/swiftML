public class GradientBoost {
    
    public var data : [[String]]
    public var target : Int
    public var residualData : [[String]]
    public var root : Float
    public var trees : [DecisionTree]
    public var limit : Int
    public var learningRate : Float
    
    
    /*
     Initialize for Gradient Boost Regressor
     Parameters:
            data -> data with labels
            target -> column number for the label
            perform -> regression or classification
            till -> the number of trees
            learningRate -> a float for learning rate
    */
    public init (data: [[String]], target : Int, perform : String, till: Int, learningRate: Float) {
        self.data = data
        self.target = target
        self.limit = till
        self.root = (Float)(0.0)
        self.learningRate = learningRate

        for i in 1 ... data.count-1 {
          self.root += (Float)(data[i][target])!
        }

        self.root = self.root/(Float)(data.count-1)
        self.residualData = data
        
        for i in 1 ... data.count-1 {
          let residual = (Float)(data[i][target])!-self.root
          self.residualData[i][target] = (String)(residual)
        }

        self.trees = [DecisionTree]()
    }
  
  
    public func boost(){
        for i in 1 ... self.limit {
          let tree = DecisionTree(data: self.residualData, target: self.target, perform: "regression", using : "gini")
          self.trees.append(tree)
          self.updateResidualData()
        }
    }
  
    
    public func updateResidualData() {
      let head = self.residualData[0]
    
      for i in 1 ... self.residualData.count-1 {
          var prediction = self.root
        
          for tree in trees {
            var example = [[String]]()
            example.append(head)
            example.append(self.residualData[i])
            prediction += self.learningRate*(Float)(tree.classify(example:example))!
          }
          let residual = (Float)(self.data[i][target])!-prediction
          self.residualData[i][target] = (String)(residual)
        }

    }
  
    
    public func predict(this: [[String]]) -> String {
      var prediction = self.root
      for tree in trees {
            prediction += self.learningRate*(Float)(tree.classify(example:this))!
          }
      return (String)(prediction)
    }
    
}
