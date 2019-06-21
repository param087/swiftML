import TensorFlow


public class DecisionTree {
  
  public var originalDataSet : DataSet
  public var root : Node? 
  
  //init creates original DataSet to be stored and calls id3 to create tree
  public init (data : [[String]], target: Int) {
    self.originalDataSet = DataSet(name: "Original", data: data, target: target)
    self.root = id3(d: self.originalDataSet)
  }
  
  
  //id3 recursive method to traverse the dataset to create Decision Tree
 public func id3(d : DataSet) -> Node{
    
    let h = d.homogenous()
    
    //if all the classification are the same, creates leaf
    if h.0 {
      let node = Node(c: h.1, leaf: true)
      return node
    }
    
    //if no non-target attributes are left, creates leaf with dominant class
    if d.data[0].count == 1 {

      let f = Feature(data: d.data, column : 0)
      
     
      
      let v = f.getDominantValue()
      
      let node = Node(c: v.name, leaf: true)
      
      return node
    }
    
    //gets best feature to split on and creates a node
    let f = d.getBestFeature()
    
    let node = Node(c : f.name, leaf : false)
    
    //calls id3 on all subset DataSets for all values of the best feature
    for value in f.values {

      let data = createDataSet(feature : f, featureValue : value, data : d.data, target : d.target)   
            
      let id_node : Node = id3(d: data)
      
      node.addChild(label: value.name, node: id_node)
      
    }
   
    
    return node
    
    
  }
  
  //method that classifies a example fed into the tree
   public func classify(example: [[String]]) -> String{
    
    var currentNode : Node = self.root!
    
    //loop continues till leaf is found
    while !currentNode.isLeaf {
      
      let featureName = currentNode.classification
      
      let featureCol = getColumnNumber(colName: featureName, data: example)
      
      let value = example[1][featureCol]
      
      var newNode = false
      
      for branch in currentNode.branches {
        if branch.label == value {
          currentNode = branch.to
          newNode = true
        }
      }
      
      if !newNode {
        return "unknown categorical variable" 
      }
    }
    
    return currentNode.classification
    
  }
  
  
}

//class defining a branch between two nodes
public class Branch {
  
  public var label : String
  public var from : Node
  public var to : Node
  
  public init(label: String, from: Node, to: Node){
    self.label = label
    self.from = from
    self.to = to
  }
  
}


//class defining a node in the decision tree
public class Node {
  public var classification: String
  public var isLeaf: Bool
  public weak var parent: Node?
  public var branches = [Branch]()

  public init(c: String, leaf : Bool) {
    self.classification = c
    self.isLeaf = leaf
  }

//adds a child by linking two nodes with a branch
  public func addChild(label: String, node: Node) {
    let b = Branch(label: label, from: self, to: node)
    node.parent = self 
    self.branches.append(b)
  }
}


//DataSet class to store data and perform computations on it
  public class DataSet{
    
    public var name: String
    public var data: [[String]]
    public var entropy: Float
    public var infoGains: Dictionary<Feature, Float>
    public var splitFeature: Feature
    public var target: Int
    
    public init(name: String, data: [[String]], target: Int){
        self.name = name
        self.data = data 
        self.entropy = 0.0
        self.target = target
        self.infoGains = Dictionary<Feature, Float>()
        if target != 0 {
          self.splitFeature = Feature(data: self.data, column : 0)    
        } else {
          self.splitFeature = Feature(data: self.data, column : 1)    
        }
      getEntropy()
    }
    
    //returns if dataset has same target classification for all examples
    public func homogenous() -> (Bool, String) {
      
      let classification : String = self.data[1][self.target]
      
      for i in stride(from: 1, through: data.count-1, by:1){
        if self.data[i][self.target] != classification {
          return (false, classification)
        }
      }
      
      return (true, classification)
    }
    
    //returns the entropy of the data set
    public func getEntropy() -> Float{
      
      let t : Feature = Feature(data: self.data, column: self.target)
      
      var e : Float = 0.0
      
      let total = self.data.count - 1
      
      for value in t.values {
        let number = (Float)(value.occurences)/(Float)(total)
        e += -1 * number * log2(number)
      }
      
      self.entropy = e
      
      return e
      
    }
    
    //methods returns the bestFeature i.e. max infoGain 
    public func getBestFeature() -> Feature {
      
      var bestGain : Float = self.splitFeature.getInfoGain(data: self.data, target: self.target)
      
      for i in stride(from: 0, through: self.data[0].count-1, by: 1){
        
        if i != self.target {
          
          let f : Feature = Feature(data: self.data, column: i)
          
          let gain : Float = f.getInfoGain(data: self.data, target: self.target)
          
          let infoGain : Float = self.entropy - gain
          
          if infoGain > bestGain {
            bestGain = infoGain
            self.splitFeature = f
          }

        }
        
      }
      
      return self.splitFeature
      
      
    }
    

    
  }
  
    
  //Helper function to delete a column in a given 2D array
  public func deleteColumn(data: [[String]], column: Int) -> [[String]] {
      
        var modified : [[String]]
        modified = []
        
      for i in stride(from: 0, through: data.count-1, by: 1){
         var d : [String]
         d = []
         
         for j in stride(from: 0, through: data[i].count-1, by: 1){
           
            if j == column {
                continue
             }
           d.append(data[i][j])
             }
  
         modified.append(d)
          }

        return modified
     }
    
    /*
    returns the column number for a given column name
    used to find a feature's corresponding column    
    */
    public func getColumnNumber(colName: String, data: [[String]]) -> Int{
      var col = -1 
      
      for i in stride(from: 0, through: data[0].count-1, by: 1){
        if data[0][i] == colName{
          col = i
          return col
        }
      }
      
      return col
      
    }
    
    //creates a subset DataSet where all examples have a specific feature value
    public func createDataSet(feature : Feature, featureValue : FeatureValue, data : [[String]], target : Int) -> DataSet{
    
       let c = getColumnNumber(colName : feature.name, data: data)
      
       var mod = [[String]]() 
      
       mod.append(data[0])
      
       for i in stride(from: 1, through: data.count-1, by: 1){
         if data[i][c] == featureValue.name {
           mod.append(data[i])
         }
       }
      
       let name = feature.name + ":" + featureValue.name
      
       let d : [[String]] = deleteColumn(data: mod, column: c)
       let targetName : String = data[0][target]
       let t : Int = getColumnNumber(colName: targetName, data: d)
      
       let dataSet = DataSet(name: name, data: d, target : t)
      
       return dataSet
       
    
    }
  

  
  //class defining a Feature and performing computations on it
  public class Feature : Hashable{
    
    public var name : String
    public var values : Set<FeatureValue>
    public var entropy : Float
    
    public init(data : [[String]], column : Int){
      self.name = data[0][column]
      self.values = Set<FeatureValue>()
      self.entropy = 0.0
      for i in stride(from: 1, through: data.count-1, by: 1) {
        let val = data[i][column]
        let v = FeatureValue(name: val, occurences: 0)
        
        if(values.contains(v)){
          let removed = self.values.remove(v)!
          let add = FeatureValue(name: val, occurences: removed.occurences+1)
          self.values.insert(add)
        } else {
          let add = FeatureValue(name: val, occurences: 1)
          self.values.insert(add)
        }
        
      }
      
    }
    
    public func getDominantValue() -> FeatureValue{
     
      var dominantV = values.first!
      
      for v in values {
        
        if v.occurences > dominantV.occurences {
          dominantV = v
        }
        
      }
      
      return dominantV
      
    }
    
    
    public func getInfoGain(data: [[String]], target : Int) -> Float{
      
      var i : [Float] = []
      
      let total = data.count-1
      
      for v in self.values {
        
        let d : DataSet = createDataSet(feature : self, featureValue : v, data : data, target : target)
        
        let e : Float = d.entropy
        
        let number : Float = (Float)(v.occurences)/(Float)(total)
        
        i.append(number*e)
        
      }
      
      var infoGain : Float = 0.0
      
      for info in i {
        infoGain += info
      }
      
      return infoGain
      
    }
    
    public static func == (lhs : Feature, rhs : Feature) -> Bool{
      
      if lhs.name != rhs.name {
        return false
      }
      
      return true
      
    }
    
    public func hash(into hasher: inout Hasher) {

    }
    
  }
  
  //class defining a Feature Value and performing computations on it
  public class FeatureValue : Hashable{
    
    public var name : String
    public var occurences: Int
    
    
    public init(name: String, occurences: Int){
      self.name = name
      self.occurences = occurences
    }
    
    public static func == (lhs : FeatureValue, rhs : FeatureValue) -> Bool{
      
      if lhs.name != rhs.name {
        return false
      }
      
      return true
      
    }
    
    public func hash(into hasher: inout Hasher) {

    }
    
    
    
  }