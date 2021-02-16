---
layout: post
title: A Complete Example for C++ Reflection  
date: 2020-10-05 04:10 -0400
description: >
  Explaination of the C++ Reflection technique
image:
  path: "/assets/img/blog/tree-reflection.jpg"
related_posts: []
---

Static/Runtime Type deduction in C++ is always a difficult topic. It is a subject I think all C++ programmers should have knowledge of. Reflection is a subset of the techniques that heavily utilize type deduction. It is a very powerful solution used for cross languages compiling or cross devices communication. Some common use cases such as: XML Declarative UI framework, YAML Configuration script, JSON data interpretation, etc.

I have once needed to implement a C++ real-time data-binding synchronization framework on top of cloud service providers like Google Firebase. One big requirement is that I need to support dynamic class construction. Since the data sent are serialization of the objects at the sender side, I need to have a proper way to deserialize the string into class instances on the receiver side. The initial idea I had was using C++ reflection. After some search online I found this [A Flexible Reflection System in C++: Part 1](https://preshing.com/20180116/a-primitive-reflection-system-in-cpp-part-1/) blog, which is very clear on describing the technique and provides minimal example code. So my learning is mainly based on this blog. However, the author only talked about the serialization but didn’t touch the deserialization. **My blog below will talk about how to add the deserialization in reflection on top of the original post’s sample code.**

Although eventually, I didn’t choose reflection as the final solution for the sake of the risk and performance in the production code, the learning experience did inspire me a lot. And I am more comfortable with C++ type support features. 

* toc
{:toc .large-only}

### C++ Macro Basic
Before we heads to the reflection, let’s review the some basic of C++ Macros:

```c++
// #name convert the literal text into a string constant
#define STRINGIFY(name) #name

// a##b concatenate 2 texts together into 1 C++ code text 
#define int(a) i##n##t a=1##5;
// ## cannot direct concatenate with operator characters, like 5##-1 is invalid
// if has space around the ##, then the spaces are ignored

// Add comment within a Macro 
#define FOO(){              \
  /* Hello */ 	    \
}

// undef
#undef A

// Pass to next Macro
#define foo(N) \
  std::string passedName = N;
#define bar() \
  std::cout << passedName << std::endl;

// Forward variadic variables
#define PRINT(...) printf(__VA_ARGS__)

// Empty variadic, required C++2a
// this can handle some cases that doesn't allow comma after the last element
#define CLASS(name, ...) class name __VA_OPT__(:) __VA_ARGS__ {};
CLASS(Bar)
CLASS(Foo, public Bar)

// Macro forwarding
#define DATA_WRAP() DATA
// if there is a space between ##, then will forward the macro content
LANG ## _VEC2_ARR

// stringify a marco content
#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

// escape special char in marco
#define MACRO(hash, name) hash name hash

// For more basic info you can visit at: 
// [Macros (The C Preprocessor)](https://gcc.gnu.org/onlinedocs/cpp/Macros.html#Macros)
// [C++ Standard](https://eel.is/c++draft/cpp.rescan)
```


### C++ Macro and template advanced
The above code snippets can already help us do some interesting Meta Programming. However, in some advanced use cases, we may need more than just play with the literal text. We need the type information. Here are some examples:

```c++
// lvalue variable assignment in Macro
#define COPY(a, b) auto b = a;

// Cannot deduct type
#define COPY_FUNC(a) decltype(a) copy() { return a; } 

// Static Register
class Register {
	Register(std::string name) {
	  Registry::sRegister(name);
	}
};
Register foo("foo");

// Type info capture in lambda
template <typename T>
void foo() {
  auto functor = [=](){
  	T val = static_cast<T>(bar());
  }
}

// Container inner type deduction
template <typename T>
class TypeResolver<std::vector<T>> {
	// T is the container inner type
}

// Test if a class/struct contain a specific static member called Reflection
template <typename T>
char func(decltype(&T::Reflection));
template <typename T>
int func(...);
template <typename T>
struct IsReflected {
	enum {
		value = (sizeof(func<T>(nullptr)) == sizeof(char))
	};
};
// then we can test with IsReflected<T>::value
// When we just use sizeof(func()) to obtain the returned variable type size, 
// not actually called the func(), then we don't need to define the function.
// At compile time, when calling func<T>(nullptr) and T has the static member 
// called Reflection, then compiler will match the first definition of the func. 
// So the returned type is char. So the value is assigned true.
// Also the reason of using enum is because we can call it as IsReflected<T>::value
// Or we can also declare value as a static bool
```

### Recap the Original C++ Reflection post
Now we have the prerequisite knowledge for this blog, let’s review how the serialization works in the Original post [A Flexible Reflection System in C++: Part 1](https://preshing.com/20180116/a-primitive-reflection-system-in-cpp-part-1/).

The core idea of this implementation of Reflection is the `TypeDescriptor`, which is per type helper class for serialization. No matter the type is a C++ primitive type like int, bool, float. Or the container type like struct, vector, shared_ptr. We will define each TypeDescriptor for all the data types. And each TypeDescriptor will have 1 core API called `TypeDescriptor::dump()`, which can serialize the object itself and the objects under it into a string. There is a fundamental difference between the primitive TypeDescriptor and the container TypeDescriptor: if we view the whole data structure as a node tree, the primitive typed data are the leaf node, while the container typed data are the branch node, whose children nodes are the data it contains. So when serialize a container typed node, we need to recursively serialize all its children nodes. When serializing the whole node tree, we need to do a traversal of the tree from the root node. Here since we need to maintain the depth relationship of the nodes, we are using the Depth-first search to traversal. The logic is inside the `dump` implementation of each container TypeDescriptor. Also for each leaf node primitive data type, we need to implement `dump` to perform the serialization of the specific typed data. So in all, **the original sample code does a recursive traversal on a data node tree with pre-defined TypeDescriptors to serialize each node. The declaration of the static pre-defined TypeDescriptors is using C++ Macros.**

Input:
```c++
Node node = {
    "apple",
    5,
    std::unique_ptr<Node>{new Node{
        "banana",
        7,
        std::unique_ptr<Node>{new Node{
            "cherry",
            11,
            nullptr
        }}
    }}
};
```

Output:
```
// readable serialization
Node {
    key = std::string{"apple"}
    value = double{5}
    next = std::unique_ptr<Node>{
        Node {
            key = std::string{"banana"}
            value = double{7}
            next = std::unique_ptr<Node>{
                Node {
                    key = std::string{"cherry"}
                    value = double{11}
                    next = std::unique_ptr<Node>{nullptr}
                }
            }
        }
    }
}
```

We can see that this results in a beautifully formatted string of the input node tree. You can find the original sample code here: [GitHub - preshing/FlexibleReflection at part2](https://github.com/preshing/FlexibleReflection/tree/part2)

In the next section, I am going to talk about how to convert this formatted string back to node tree itself. 

### Problem Definition
Let’s first define our problem here. Assume we have 2 different data structs:
```c++
struct Subnode {
    bool flag;
    float value;
    std::vector<Node> siblings;
    std::vector<Subnode> subsubnode;
};

struct Node {
    std::string key;
    int value;
    std::shared_ptr<Subnode> subnode;
    std::vector<Node> children;
};
```

And we have a data node tree which contain mixture of the 2 data struct: 
```text
Node node = {"apple", 3, std::make_shared<Subnode>(subnode1), {
                {"banana", 125, nullptr, {
                    {"Hello", 15, std::make_shared<Subnode>(subnode2), {}}
                  }
                },
                {"cherry", 11, nullptr, std::vector<Node>(Node{})},
                {"C++ is a general-purpose programming "
                "language created by Bjarne Stroustrup as"
                " an extension of the C programming language, "
                "or C with Classes.", 131, nullptr, std::vector<Node>(Node{})}
              }
            };
```

We want to have a mechanism to serialize the node tree into a string and deserialize the string back the node tree. 

To test if the mechanism is correct we need to define the criteria. However, there is no direct way to compare between 2 complex node tree structures. So here I compare the string instead:
```c++
std::string serialized = serialize(node);
if(serialized == serialize(deserialize(serialized))) {
	cout << "Pass" << endl;
}
```

In order to do deserialization, I add a new pure virtual interface in the TypeDescriptor: 
```c++
/**
 * Instantiate the typed object from string
 * The obj is already allocated before and the data 
 * only contain the string serialization under this node
 */
virtual void fulfill(void* obj, const std::string& data, int indentLevel = 0) const = 0;
```

Each TypeDescriptor should override this method to properly recover the content using the string data.

### Static TypeDescriptor
The first step is generating the static TypeDescriptor for the 2 structs and the data type they contain.

```c++
struct Node {
    std::string key;
    int value;
    std::shared_ptr<Subnode> subnode;
    std::vector<Node> children;
    REFLECT()       // Enable reflection for this type
};

REFLECT_STRUCT_BEGIN(Node)
REFLECT_STRUCT_MEMBER(key)
REFLECT_STRUCT_MEMBER(value)
REFLECT_STRUCT_MEMBER(subnode)
REFLECT_STRUCT_MEMBER(children)
REFLECT_STRUCT_END()
```

The above meta programming code generates a `static reflect::TypeDescriptor_Struct Reflection` member inside the `struct Node`. And we can access to this static member using `reflect::TypeResolver<Node>::get()`. The auto registration is triggered by the ctor of the `static reflect::TypeDescriptor_Struct Reflection`.

So at this point, we have a list of member variables’ TypeDescriptor that `Node` contained. When calling `TypeDescriptor_Struct::dump`, we will iterate through all the member variables’ TypeDescriptor and recursively dump to strings. 

The serialization result of the above node tree is:
```
// readable serialization
Node {
    key = string{apple}
    value = int{3}
    subnode = std::shared_ptr<Subnode>{
        Subnode {
            flag = bool{1}
            value = float{1.2345}
            siblings = std::vector<Node>{
                [0] Node {
                    key = string{orange}
                    value = int{25}
                    subnode = std::shared_ptr<Subnode>{}
                    children = std::vector<Node>{}
                }
            }
            subsubnode = std::vector<Subnode>{}
        }
    }
    children = std::vector<Node>{
        [0] Node {
            key = string{banana}
            value = int{125}
            subnode = std::shared_ptr<Subnode>{}
            children = std::vector<Node>{
                [0] Node {
                    key = string{Hello}
                    value = int{15}
                    subnode = std::shared_ptr<Subnode>{
                        Subnode {
                            flag = bool{0}
                            value = float{4.3219}
                            siblings = std::vector<Node>{}
                            subsubnode = std::vector<Subnode>{
                                [0] Subnode {
                                    flag = bool{1}
                                    value = float{7.234}
                                    siblings = std::vector<Node>{}
                                    subsubnode = std::vector<Subnode>{}
                                }
                            }
                        }
                    }
                    children = std::vector<Node>{}
                }
            }
        }
        [1] Node {
            key = string{cherry}
            value = int{11}
            subnode = std::shared_ptr<Subnode>{}
            children = std::vector<Node>{
                [0] Node {
                    key = string{}
                    value = int{0}
                    subnode = std::shared_ptr<Subnode>{}
                    children = std::vector<Node>{}
                }
            }
        }
        [2] Node {
            key = string{C++ is a general-purpose programming language created by Bjarne Stroustrup as an extension of the C programming language, or C with Classes.}
            value = int{131}
            subnode = std::shared_ptr<Subnode>{}
            children = std::vector<Node>{
                [0] Node {
                    key = string{}
                    value = int{0}
                    subnode = std::shared_ptr<Subnode>{}
                    children = std::vector<Node>{}
                }
            }
        }
    }
}
```
Next we will convert the above string to a node tree.

### Deserialization
As we see from the above string, we have a lot of extra meta info besides the data itself. So for deserialization of each node, the first step is culling the extra text off.

Here since the formatted string is always aligned depth with indents and the actual data is always wrapped by the `{}`. So under this assumption, we can retrieve the data content of each node. 

This is the implementation of the `TypeDescriptor_Struct::fulfill`:
```c++
virtual void fulfill(void* obj, const std::string& data, int indentLevel) const override {
  // obj here is already allocated
  std::string indent = "\n" + std::string(4 * (indentLevel + 1), ' ');
  std::string data_ = GetRootContent(data);
  size_t curNeedle = data_.find(FormatStr("%s%s = %s", indent.c_str(), members.front().name, members.front().type->getFullName().c_str())) + 1;
  for(size_t i = 0; i < members.size(); ++i){
    size_t nextNeedle;
    if(i == members.size() - 1){
      nextNeedle = data_.size();
    }else{
      nextNeedle = data_.find(FormatStr("%s%s = %s", indent.c_str(), members[i+1].name, members[i+1].type->getFullName().c_str())) + 1;
    }
    // find between cur - next        
    std::string content = GetRootContent(data_.substr(curNeedle, nextNeedle-curNeedle));
    members[i].type->fulfill((char*) obj + members[i].offset, content, indentLevel + 1);
    curNeedle = nextNeedle;
  }
}
```

We will only search the corresponding depth level within the string and the `GetRootContent ` can return the most outside level bracket wrapped content. For each member, we will find the pointer address to the beginning of the member, which is `(char*) obj + members[i].offset`. Also we extract the content of this member using `GetRootContent(data_.substr(curNeedle, nextNeedle-curNeedle));`. Then increment the indentLevel and passing all these info to `members[i].type->fulfill`. So we start recursively deserialize the node tree. 

Here we also support different type of struct as data member, as long as the the struct is reflected. Notice that we are only using struct container type as the root node type. So in this `fulfill`, we didn’t instantiate the actual node structure or each members. Because they are already declared in the `main.cpp`. However, for other container type, we need to allocate memory for the data item it contains. For example, this is the std::vector<T>’s fulfill implementation:

```c++
virtual void fulfill(void* obj, const std::string& data, int indentLevel) const override {
  if(data.empty()){
    instantiate(obj, 0);
  }else{
    std::string indent = "\n" + std::string(4 * (indentLevel + 1), ' ');
    std::vector<std::string> items;
    size_t lastPos = 0;
    size_t count = 0;
    size_t pos;
    while((pos = data.find(FormatStr("%s[%d] %s", indent.c_str(), count, itemType->getFullName().c_str()))) 
              != std::string::npos){
      if(count) items.push_back(data.substr(lastPos, pos-lastPos));
      lastPos = pos;
      count++;
    }
    items.push_back(data.substr(lastPos));        
    instantiate(obj, items.size());
    for(size_t i = 0; i < items.size(); ++i){
      itemType->fulfill(getRawItem(obj, i), items[i], indentLevel+1);
    }
  }
}


// todo: use the template lambda with c++20
auto instantiate = [](void*& obj, size_t sz) -> void{
  auto& vec = *(std::vector<ItemType>*)obj;
  vec.resize(sz);
};
auto getRawItem = [](void* vecPtr, size_t index) -> void* {
    auto& vec = *(std::vector<ItemType>*) vecPtr;
    return &vec[index];
};

```

`instantiate` is the lambda to allocate the vector of item type with the data size. `getRawItem` is the lambda to return the raw pointer address of each item. Notice that here we use lambda to capture the template type. So that we can use it as a specialization template lambda. C++2a will introduce the template lambda, which allows us to pass the type info when calling the lambda. It should be a better solution.

`TypeDescriptor_shared_ptr` is similar to the implementation for vector.

### Primitive Type dump and fulfill
By far, we can successfully redirect from the container branch node. However, we still haven’t touched the actual data. Once we reach the leaf level, all the data will be a primitive or empty container. So here we should implement the fulfill for primitive type. 

Notice that on the above serialization example, I was using read-able serialization for better understanding. However, when we want to deserialize, we have to serialize the data to the arbitrary bytes array or base64 encoded string. 

Also we have a lot of primitive types to support. So I use Macros again to auto generate them:
```c++
#define METAPROGRAMMING(type)\
struct TypeDescriptor_##type : TypeDescriptor {\
  TypeDescriptor_##type() : TypeDescriptor{#type, sizeof(type)} {\
  }\
  void dump(const void* obj, std::stringstream& ss, bool readable, int /* unused */) const override {\
    /* Convert to byte array, not human readable */\
    if(readable){\
      ss << #type << "{" << *(const type*)obj << "}";\
    }else{\
      if(std::is_same<type, string>::value){\
        ss << #type << "{" << *(const string*)obj << "}";\
      }else{\
        auto p = reinterpret_cast<const char*>(obj);\
        ss << #type << "{" << string(p, sizeof(type)) << "}";\
      }\
    }\
  }\
  void fulfill(void* obj, const std::string& data, int /* unused */) const override{\
    if(std::is_same<type, string>::value){\
      *(string*)obj = data;\
    }else{\
      *(type*)obj = ParseAs<type>(data);\
    }\
  }\
};\
template<>\
TypeDescriptor* getPrimitiveDescriptor<type>(){\
  static TypeDescriptor_##type typeDesc;\
  return &typeDesc;\
}

METAPROGRAMMING(int)
METAPROGRAMMING(bool)
METAPROGRAMMING(float)
METAPROGRAMMING(double)
METAPROGRAMMING(char)
METAPROGRAMMING(string)
```

So when calling `reflect::TypeResolver<Primitive_Type>::get()`, there is no `Primitive_Type::Reflection` exist in the primitive type, so it is not a reflected class and we end up using the TypeDescriptors above. 

`ParseAs ` will just write the arbitrary bytes to the object’s pointer beginning address. This works fine for C++ basic primitive type. But for `std::string`, we need a specialization implementation.


### Things still can be done
So eventually after the traversal, we can obtain the node tree object from the input string. But we have made several assumptions here like the `indent` represents the node depth. So a future improvement can be using JSON format for serialization, which use has no extra meta info and use brackets to indicates depth.

### Demo Code
The final demo code can be found at [GitHub - coroner4817/FlexibleReflection at deserialization](https://github.com/coroner4817/FlexibleReflection/tree/deserialization)