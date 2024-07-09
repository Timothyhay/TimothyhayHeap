---
layout: blogpage
title: Describe a System w/ 4+1 Architectural View Model!
comments: true
tags: Note
---

## 4+1 architectural view model

From Wikipedia:

4+1 is a view model used for "describing the architecture of software-intensive systems, based on the use of multiple, concurrent views".[1] The views are used to describe the system from the viewpoint of different stakeholders, such as end-users, developers, system engineers, and project managers. The four views of the model are **logical, development, process and physical view**. In addition, selected **use cases or scenarios** are used to illustrate the architecture serving as the 'plus one' view. Hence, the model contains 4+1 views:

4+1 是一种视图模型，用于“通过使用多个并列视图来描述软件密集型系统的架构”。 这些视图用于从不同利益相关者（如最终用户、开发人员、系统工程师和项目经理）的角度描述系统。该模型的 4 个视图是**逻辑视图、开发视图、过程视图和物理视图**。此外，选定的**用例或场景**用于说明作为“+1”视图的架构。因此，该模型包含 4+1 视图： 


**Logical view:** The logical view is concerned with the functionality that the system provides to end-users. **UML diagrams** are used to represent the logical view, and *include class diagrams, and state diagrams*.

**逻辑视图**：逻辑视图关心系统向最终用户提供的功能。**UML 图**用于表示逻辑视图，*包括类图和状态图*。 


**Process view**: The process view deals with the dynamic aspects of the system, explains the system processes and how they communicate, and focuses on the run time behavior of the system. The process view addresses concurrency, distribution, integrator, performance, and scalability, etc. **UML diagrams** to represent process view *include the sequence diagram, communication diagram, activity diagram.*

**流程视图**：流程视图处理系统的动态方面，解释系统进程及其通信方式，并关注系统的运行时行为。流程视图涉及并发性、分布式、集成器、性能和可扩展性等。表示流程视图的 UML 图*包括序列图、通信图、活动图*。


**Development view**: The development view (aka the implementation view) illustrates a system from a programmer's perspective and is concerned with software management. **UML Diagrams** used to represent the development view *include the Package diagram and the Component diagram*.

**开发视图**：开发视图（又名实现视图）从程序员的角度说明系统，并关注软件管理。用于表示开发视图的 **UML图** *包括包关系图和组件关系图*。 


**Physical view**: The physical view (aka the deployment view) depicts the system from a system engineer's point of view. It is concerned with the topology of software components on the physical layer as well as the physical connections between these components. **UML diagrams** used to represent the physical view *include the deployment diagram*.

**物理视图**：物理视图（也称为部署视图）从系统工程师的角度描述系统。它涉及物理层上软件组件的拓扑结构以及这些组件之间的物理连接。用于表示物理视图的 **UML 图**包括*部署关系图*。 


**Scenarios**: The description of an architecture is illustrated using a small set of use cases, or scenarios, which become a fifth view. The scenarios describe sequences of interactions between objects and between processes. They are used to identify architectural elements and to illustrate and validate the architecture design. They also serve as a starting point for tests of an architecture prototype. This view is also known as the use case view.

**场景**：使用一小组用例或场景来说明架构的描述，这些用例或场景成为第五个视图。这些场景描述了对象之间和进程之间的交互序列。它们用于识别架构元素，并说明和验证架构设计。它们还可以作为架构原型测试的起点。此视图也称为用例视图。


The 4+1 view model is generic and is not restricted to any notation, tool or design method.

4+1 视图模型是通用的，不限于任何符号、工具或设计方法。