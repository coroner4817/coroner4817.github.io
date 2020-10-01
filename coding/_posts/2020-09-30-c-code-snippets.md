---
layout: post
title: C++ code snippets
date: 2020-09-30 03:22 -0400
categories: [coding]
description: > 
  Some C++ tricks and syntactic sugars
image:
  path: "/assets/img/blog/cpp.jpg"
related_posts: []
---

Recording some coding snippets I have learned.  

### Multiple instancing with Marco
```c++
#define INSTANCING_LIST(...) , ##__VA_ARGS__
#define DECLARE(name, ...) name INSTANCING_LIST(__VA_ARGS__)
```
The magic is in the `INSTANCING_LIST`. if `__VA_ARGS__` is empty then it return nothing, 
else it will return with a comma append at front. This can solve the issue when multiple instancing declaration doesn't allow last item end with a comma
Like the class inheritance case

A better solution is using c++20 feature `__VA_OPT__`

### Lambda capture a member variable
```c++
auto func = [val = data.float](){
  // use val
};
```

### Struct assignment
```c++
DataStruct a = {
  .f = 1.f,
  .s = ""
};
```

### Tuple return
```c++
std::tuple<Type1, Type2> func(){};
auto[data1, data2] = func();
```

### Template class separate to .inl 
```c++
// header
template <typename T, int val = 1>
class MyClass {
  void foo();
};
#inclde "class.inl"

// inl
template <typename T, int val>
MyClass<T, val>::foo(){
  // impl
}
```

### std::forward
```c++
// SFINAE
template <typename U, std::enable_if_t<std::is_same<T, std::decay_t<U>>::value, int> = 0>
void set(U&& value){
  T val = std::forward<U>(value);
}
```
Allow both lvalue and rvalue passing through same API. Along the code path, all functions needs to do std::forward
[C++ type utiles function](https://en.cppreference.com/w/cpp/types)


### Variadic arguments template args
```c++
template <typename... Args>
void func(Args&&... args) {
  callback(std::forward<Args>(args)...);
}
```

### Redirect class type
```c++
class A {
  operator B() const {
    return this->b;
  };
};
A a;
// Then a can used as B
```
