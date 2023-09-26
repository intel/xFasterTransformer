# Coding Standards

## Introduction

The Coding Standards outline and describe a set of rules the xFasterTransformer code base
and code contributions must follow.

The only goal of the Coding Standards is to maintain productivity for the
development of the library.

The Coding Standards are subject to change.

---
**NOTE** It may happen that not all code can and does follow all the rules
outlined in the Coding Standards. In case enforcing a specific rule makes your
contribution less effective in terms of code design, please do not apply the
corresponding rule.

---

### Coding style

The coding style consistency in xFasterTransformer is maintained using `clang-format`.
When submitting your contribution, please make sure that it adheres to the existing
coding style with the following command:
```sh
clang-format -style=file -i foo.cpp
```
This will format the code using the `_clang_format` file found in the xFasterTransformer
top level directory.

Coding style is secondary to the general code design.

### General

- Use properly named constants whenever possible (unless this code is
  auto-generated).
  * For example,
  ~~~cpp
  if (x < 4) y = 64;
  ~~~

  In this example, 4 and 64 should be named, in which case the code becomes:
  ~~~cpp
  if (x < sizeof(float)) y = cache_line_size;
  ~~~

- Don't use `using namespace XXX` in header files.

- Avoid code duplication (look for similarities), unless it is necessary.

- Declare variables in the innermost possible scope until there are some
  circumstances that make you declare them somewhere else.

- Consider using utils to improve readability (`IMPLICATION`, `one_of`,
  `everyone_is`).
