public class DecisionTree {
  
    public var originalDataSet : DataSet
    public var root : Node?
    public var maxDepth : Int
  
    //init creates original DataSet to be stored and creates tree
    public init (data: [[String]], target: Int, perform: String, using: String) {
        self.originalDataSet = DataSet(name: "Original", 
                                       data: data, 
                                       target: target)
        self.maxDepth = 9999 
      
        if perform == "regression" {
            if using == "infogain" {
               self.root = id3R(d: self.originalDataSet)
            } else if using == "gini" {
               self.root = giniR(d: self.originalDataSet, depth: 0)
            } else {
              print("invalid operation requested")
            }
        } else if perform == "classification" {
            if using == "infogain" {
               self.root = id3C(d: self.originalDataSet)
            } else if using == "gini" {
               self.root = giniC(d: self.originalDataSet, depth: 0)
            } else {
              print("invalid operation requested")
            }
        } else {
          print("invalid operation requested")
        }
      
    }
  
   //init creates original DataSet to be stored and creates tree WITH depth requested
    public init (data : [[String]], target: Int, maxDepth: Int, perform: String, using: String) {
        self.originalDataSet = DataSet(name: "Original", 
                                       data: data, 
                                       target: target
                                       )
        self.maxDepth = maxDepth 
        if perform == "regression" {
            if using == "infogain" {
                self.root = id3R(d: self.originalDataSet)
            } else if using == "gini" {
                self.root = giniR(d: self.originalDataSet, depth: 0)
            } else {
                print("invalid operation requested")
            }
        } else if perform == "classification" {
            if using == "infogain" {
                self.root = id3C(d: self.originalDataSet)
            } else if using == "gini" {
                self.root = giniC(d: self.originalDataSet, depth: 0)
            } else {
                print("invalid operation requested")
            }
        } else {
            print("invalid operation requested")
        }
    }
    
    //method that prints the Decision Tree
    public func printTree(node: Node, depth : Int){
        var indent : String = ""

        for x in 0 ... depth {
            indent += "  "
        }
    
        print(indent + node.classification)
    
        print("\n")
    
        if (!node.isLeaf){
            for b in node.branches {
                print(indent + b.label)
                printTree(node : b.to, depth: depth+1)
            }
         }
    } 
  
    
    //Method that forms decision regression tree using gini index
    public func giniR(d : DataSet, depth: Int) -> Node {
        let currentGiniImpurity = d.getGiniImpurity()
        let f = d.getGiniFeature()

        if f.giniImpurity < currentGiniImpurity && depth <= self.maxDepth {
            let node = Node(c : f.name, leaf : false)

            if (Float)(f.values.first!.name) == nil {
                for value in f.values {
                    let data = createDataSet(feature : f, 
                                     featureValue : value, 
                                     data : d.data, 
                                     target : d.target)   
            
                    let gNode : Node = giniR(d: data, depth : depth+1)
      
                    node.addChild(label: value.name, node: gNode)
                }    
                return node
            } else {
                let datas = f.giniImpurityNumerical(data: d.data, target : d.target) 

                let d1 = deleteColumn(data: datas.1.data, column: getColumnNumber(colName: f.name, data: datas.1.data))
                let d2 = deleteColumn(data: datas.2.data, column:  getColumnNumber(colName: f.name, data: datas.2.data))
                
                let data1 = DataSet(name: "one", data: d1, target: getColumnNumber(colName: d.data[0][d.target], data: d1))
                let data2 = DataSet(name: "two", data: d2, target: getColumnNumber(colName: d.data[0][d.target], data: d2))

                let leftNode : Node =   giniR(d: data1, depth : depth+1)
                let rightNode : Node =   giniR(d: data2, depth : depth+1)
            
                node.addFork(label: "more than equal "+(String)(datas.3), node: rightNode, cutoff: datas.3)
                node.addFork(label: "less than "+(String)(datas.3), node: leftNode, cutoff: datas.3)
                
                return node
            }    
            return node

        } else {
          
            let node = Node(c: String(d.getTargetMean()), leaf: true)
            return node
        
        }
      
    }
   
    //Method that forms decision classification tree using gini index
    public func giniC(d : DataSet, depth: Int) -> Node {
        let h = d.homogenous()
    
        //if all the classification are the same, creates leaf
        if h.0 {
            let node = Node(c: h.1, leaf: true)
            return node
        }
    
        //if no non-target attributes are left, creates leaf with dominant class
        if d.data[0].count <= 1 {
            let f = Feature(data: d.data, column : 0)
            let v = f.getDominantValue()
            let node = Node(c: v.name, leaf: true)
            return node
        }
        
        let currentGiniImpurity = d.getGiniImpurity()
        let f = d.getGiniFeature()

        if currentGiniImpurity != 0.0 && f.giniImpurity < currentGiniImpurity && depth < self.maxDepth {
            let node = Node(c : f.name, leaf : false)
              
            if (Float)(f.values.first!.name) == nil {
                for value in f.values {
                    let data = createDataSet(feature : f, 
                                     featureValue : value, 
                                     data : d.data, 
                                     target : d.target)   
                    let g_node : Node = giniC(d: data, depth : depth+1)
                    node.addChild(label: value.name, node: g_node)
                }    
                return node
            } else {
                let datas = f.giniImpurityNumerical(data: d.data, target : d.target)                        
 
                let d1 = deleteColumn(data: datas.1.data, column: getColumnNumber(colName: f.name, data: datas.1.data))
                let d2 = deleteColumn(data: datas.2.data, column:  getColumnNumber(colName: f.name, data: datas.2.data))
              
                let data1 = DataSet(name: "one", data: d1, target: getColumnNumber(colName: d.data[0][d.target], data: d1))
                let data2 = DataSet(name: "two", data: d2, target: getColumnNumber(colName: d.data[0][d.target], data: d2))

                let leftNode : Node =   giniC(d: data1, depth : depth+1)
                let rightNode : Node =   giniC(d: data2, depth : depth+1)
            
                node.addFork(label: "more than equal "+(String)(datas.3), node: rightNode, cutoff: datas.3)
                node.addFork(label: "less than "+(String)(datas.3), node: leftNode, cutoff: datas.3)
                
                return node
            }    
        
           return node
          
         } else {
          
            let f = Feature(data: d.data, column : d.target)
            let v = f.getDominantValue()
            let node = Node(c: v.name, leaf: true)
            return node
        }
      
    }
  
  
    //id3 recursive method to examine the dataset to create classification Tree
    public func id3C(d : DataSet) -> Node{
    
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
            let data = createDataSet(feature : f, 
                                     featureValue : value, 
                                     data : d.data, 
                                     target : d.target)   
            
            let id_node : Node = id3C(d: data)
            node.addChild(label: value.name, node: id_node)
        }

        return node
    }
  
    //id3 recursive method to traverse the dataset to create Decision Tree
    public func id3R(d : DataSet) -> Node{

        //if all the classification are the same, creates leaf
        //if no non-target attributes are left, creates leaf with dominant class
        let tolerance : Float = 10.0
        if d.getCoeffDev() < tolerance || d.data[0].count == 1 || d.data.count < 4 {
            let node = Node(c: String(d.getTargetMean()), leaf: true)
            return node
        }

        //gets best feature to split on and creates a node
        let f = d.getSplitFeature()
        let node = Node(c : f.name, leaf : false)
    
        //calls id3 on all subset DataSets for all values of the best feature
        for value in f.values {
            let data = createDataSet(feature : f, 
                                     featureValue : value, 
                                     data : d.data, 
                                     target : d.target)   
            let id_node : Node = id3R(d: data)
            node.addChild(label: value.name, node: id_node)
        }    

        return node
    }
  
    //method that classifies a example fed into the tree
    public func classify(example: [[String]]) -> String {
        var currentNode : Node = self.root!

        //loop continues till leaf is found
        while !currentNode.isLeaf {
      
            let featureName = currentNode.classification
            let featureCol = getColumnNumber(colName: featureName, data: example)
            let value = example[1][featureCol]
            var newNode = false

            if (Float)(value) == nil {
                for branch in currentNode.branches {
                    if branch.label == value {
                        currentNode = branch.to
                        newNode = true
                    }
                }
            } else {
                print(currentNode.cutoff, value)
                if (Float)(value)! <= currentNode.cutoff! { 
                    currentNode = currentNode.branches.last!.to
                    newNode = true
                } else {
                    currentNode = currentNode.branches.first!.to
                    newNode = true
                }

                if !newNode {
                    return "unknown categorical variable" 
                }
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
    public var cutoff : Float?
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
  
    public func addFork(label: String, node: Node, cutoff: Float) {
        let b = Branch(label: label, from: self, to: node)
        self.cutoff = cutoff
        node.parent = self 
        self.branches.append(b)
    }
}


//DataSet class to store data and perform computations on it
public class DataSet {
    
    public var name: String
    public var data: [[String]]
    public var entropy: Float
    public var infoGains: Dictionary<Feature, Float>
    public var splitFeature: Feature
    public var target: Int
    public var stdDev: Float
    public var giniImpurity : Float
    
    public init(name: String, data: [[String]], target: Int){
        self.name = name
        self.data = data 
        self.entropy = 0.0
        self.stdDev = 0.0
        self.target = target
        self.infoGains = Dictionary<Feature, Float>()
        self.giniImpurity = 0.0
        if target != 0 {
            self.splitFeature = Feature(data: self.data, column : 0)    
        } else if data[0].count > 1{
            self.splitFeature = Feature(data: self.data, column : 1)    
        } else {
            self.splitFeature = Feature(data: self.data, column : 0)    
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
  
    public func getCoeffDev() -> Float {
        return (getTargetStdDev()/getTargetMean())*100
    }
  
    public func getTargetStdDev() -> Float{
        let t : Feature = Feature(data: self.data, column: self.target)
        var sd : Float = 0.0
        let total = self.data.count - 1
        let mean = getTargetMean()
        var s : Float = 0.0
      
        for value in t.values {
            let number = (Float)(value.name)!-mean
            s += (pow(number, 2))*(Float)(value.occurences)
        }
      
        sd = Float((s/(Float)(total)).squareRoot())
        return sd 
    }
  
    public func getTargetMean() -> Float{
        let t : Feature = Feature(data: self.data, column: self.target)
        var mean : Float = 0.0
        let total = self.data.count - 1
        var count : Float = 0.0

        for value in t.values {
            count += (Float)(value.name)!*(Float)(value.occurences)
        }

        mean = count/(Float)(total)
        return mean
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
    
    //method returns the bestFeature i.e. max infoGain 
    public func getBestFeature() -> Feature {
        var bestInfoGain : Float = self.splitFeature.getInfoGain(data: self.data, 
                                                              target: self.target)
        var bestGain : Float = self.entropy - bestInfoGain
      
        for i in stride(from: 0, through: self.data[0].count-1, by: 1){
            if i != self.target {
                let f : Feature = Feature(data: self.data, column: i)
                let gain : Float = f.getInfoGain(data: self.data, 
                                                 target: self.target)
                let infoGain : Float = self.entropy - gain
          
                if infoGain > bestGain {
                    bestGain = infoGain
                    self.splitFeature = f
                }

            }
        
        }
      
        return self.splitFeature
    }
  
    //method returns best gini feature i.e. minimum gini impurity
    public func getGiniFeature() -> Feature {
      
        var bestImpurity : Float 
       
        if (Float)(self.splitFeature.values.first!.name) == nil {
           bestImpurity = self.splitFeature.giniImpurityCategorical(data: self.data, 
                                                                    target : self.target) 
        } else {
           bestImpurity = self.splitFeature.giniImpurityNumerical(data: self.data, 
                                                                    target : self.target).0 
        }
        
      
        for i in stride(from: 0, through: self.data[0].count-1, by: 1){
            if i != self.target {
                let f : Feature = Feature(data: self.data, column: i)
                var impurity : Float

                if (Float)(f.values.first!.name) == nil {
                   impurity = f.giniImpurityCategorical(data: self.data, 
                                                            target : self.target) 
                } else {
                   impurity = f.giniImpurityNumerical(data: self.data, 
                                                          target : self.target).0 
                }
          
                if bestImpurity > impurity {
                    bestImpurity = impurity
                    self.splitFeature = f
                }
            }
        }
        return self.splitFeature
    }
   
   
    //method returns feature with most standard deviation reduction
    public func getSplitFeature() -> Feature {
        var bestSD : Float = self.splitFeature.getTargetStdDev(data: self.data, 
                                                               target: self.target)
        var bestSDR : Float = getTargetStdDev() - bestSD
      
        for i in stride(from: 0, through: self.data[0].count-1, by: 1){
            if i != self.target {
                let f : Feature = Feature(data: self.data, column: i)
                let sdt : Float = f.getTargetStdDev(data: self.data, 
                                                 target: self.target)
                let sdr : Float = getTargetStdDev() - sdt
          
                if sdr > bestSDR {
                    bestSDR = sdr
                    self.splitFeature = f
                }
            }
        }
        return self.splitFeature
    }
  
    //returns gini impurity of the data set
    public func getGiniImpurity() -> Float {
        let t : Feature = Feature(data: self.data, column: self.target)
        var i : Float = 0.0
        let total = self.data.count - 1
      
        for value in t.values {
            let number = (Float)(value.occurences)/(Float)(total)
            i += pow(number,2)
        }

        self.giniImpurity = 1-i
        return 1-i
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
public func createDataSet(feature : Feature, 
                          featureValue : FeatureValue, 
                          data : [[String]], 
                          target : Int) -> DataSet{
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

public func splitDataSet(data: [[String]], startIndex: Int) -> ([[String]], [[String]]) {
    let title = data[0]
    var firstPart = Array(data[1..<startIndex])
    var secondPart = Array(data[startIndex..<data.count])
    
    firstPart.insert(title, at:0)
    secondPart.insert(title, at:0)
    return(firstPart,secondPart)
}
  

  
//class defining a Feature and performing computations on it
public class Feature : Hashable {
    
    public var name : String
    public var values : Set<FeatureValue>
    public var entropy : Float
    public var giniImpurity : Float
    
    public init(data : [[String]], column : Int){
        self.name = data[0][column]
        self.values = Set<FeatureValue>()
        self.entropy = 0.0
        self.giniImpurity = 0.0
     
        for i in stride(from: 1, through: data.count-1, by: 1) {
            let val = data[i][column]
            let v = FeatureValue(name: val, occurences: 0)
        
            if(values.contains(v)){
                let removed = self.values.remove(v)!
                let add = FeatureValue(name: val, 
                                       occurences: removed.occurences+1)
                self.values.insert(add)
            } else {
                let add = FeatureValue(name: val, occurences: 1)
                self.values.insert(add)
            }
        
        }
      
    }
  
    //returns most occuring featureValue
    public func getDominantValue() -> FeatureValue{
        var dominantV = values.first!
    
        for v in values {
            if v.occurences > dominantV.occurences {
            dominantV = v
            }
        }
      return dominantV
    }
  
  
    public func getTargetStdDev(data: [[String]], target : Int) -> Float{
        var i : [Float] = []
        let total = data.count-1
        
        for v in self.values {
            let d : DataSet = createDataSet(feature : self, 
                                            featureValue : v, 
                                            data : data, 
                                            target : target)
            let sdt : Float = d.getTargetStdDev()
            let number : Float = (Float)(v.occurences)/(Float)(total)
            i.append(number*sdt)
        }
        
        var sd : Float = 0.0
        for info in i {
            sd += info
        }
        return sd
    }
    
    //returns infoGain for the feature
    public func getInfoGain(data: [[String]], target : Int) -> Float{
        var i : [Float] = []
        let total = data.count-1
      
        for v in self.values {
        
            let d : DataSet = createDataSet(feature : self, 
                                            featureValue : v, 
                                            data : data, 
                                            target : target)
        
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
  
    //returns categorical gini impurity for the feature
    public func giniImpurityCategorical(data: [[String]], target : Int) -> Float {
       var giniImp : Float = 0.0
       let total = data.count-1
        
       for value in self.values {
           let d : DataSet = createDataSet(feature : self, 
                                            featureValue : value, 
                                            data : data, 
                                            target : target)
        
           let i : Float = d.getGiniImpurity()
       
           giniImp += i*((Float)(value.occurences)/(Float)(total))
       }
       self.giniImpurity = giniImp
       return giniImp
    }
                                         
      
    public func giniImpurityNumerical(data: [[String]], target : Int) -> (Float, DataSet, DataSet, Float) {
        let title = data[0]
        var mod = Array(data[1...data.count-1])
        let col : Int = getColumnNumber(colName: self.name, data: data)
                   
        mod.sort { left, right in
                 (Float)(left[col])! < (Float)(right[col])!
               }
        mod.insert(title, at:0)

        let total = data.count-1
        var split = splitDataSet(data: mod, startIndex: 2)
        var data1 : DataSet = DataSet(name: "one", data: split.0, target: target)
        var data2 : DataSet = DataSet(name: "two", data: split.1, target: target)
        var giniI = max(data1.getGiniImpurity(), data2.getGiniImpurity())
       
       
        for i in stride(from: 2, through: data.count-1, by: 1){
            split = splitDataSet(data: mod, startIndex: i)
         
            let d1 : DataSet = DataSet(name: "one", data: split.0, target: target)
            let d2 : DataSet = DataSet(name: "two", data: split.1, target: target)
            let impurity = max(d1.getGiniImpurity(), d2.getGiniImpurity())

            if impurity < giniI {
                giniI = impurity
                data1 = d1
                data2 = d2
            } 
        }
      
        self.giniImpurity = giniI
        return (giniI, data1, data2, (Float)(data1.data.last![getColumnNumber(colName: self.name, data: data1.data)])!)
    }
    
    //required == func for implementing Hashable
    public static func == (lhs : Feature, rhs : Feature) -> Bool{
        if lhs.name != rhs.name {
          return false
        }
        return true
    }
    
    //empty required hash func for implementing Hashable
    public func hash(into hasher: inout Hasher) {
    }
    
}
  
  //class defining a Feature Value and performing computations on it
public class FeatureValue : Hashable {
    
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



                            
