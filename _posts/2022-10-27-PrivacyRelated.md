---
layout: blogpage
title: Security & Privacy Bullet pts!
comments: true
tags: Note
---

**ASLR**（Address space layout randomization）是一种针对缓冲区溢出的安全保护技术，通过对堆、栈、共享库映射等线性区布局的随机化，通过增加攻击者预测目的地址的难度，防止攻击者直接定位攻击代码位置，达到阻止溢出攻击的目的。

**DPIA** - GDPR 要求，数据控制者必须准备对“可能会对自然人的权利和自由造成高风险”的操作执行数据保护影响评估 (DPIA)。 

**CFI** - 为了抵御控制流劫持攻击，Abadi等人于2005年提出了控制流完整性（Control Flow Integrity, CFI）的防御机制，是一种通过保证控制流的完整性来实现软件防篡改的方法（基于插桩技术）。其核心思想是限制程序运行中的控制转移，使之始终处于原有的控制流图所限定的范围内。它规定软件执行必须遵循提前确定的ControlFlow图（CFG）的路径。

该方法虽能防范多种类型攻击，但实现复杂，且由于判断软件是否被篡改的全部逻辑都在该软件自身中存在，故该防护机制易于被攻击者发现和利用。

**STRIDE** 是从攻击者的角度，把威胁划分成 6 个类别，分别是 Spooling（仿冒）、Tampering(篡改)、Repudiation（抵赖）、InformationDisclosure（信息泄露）、Dos(拒绝服务) 和 Elevation of privilege (权限提升)。

| 威胁          | 安全属性 | 定义                   | 举例                 |
|---------------|----------|------------------------|----------------------|
| 仿冒（S）     | 认证     | 冒充人或物             | 冒充其他用户账号     |
| 篡改（T）     | 完整性   | 修改数据或代码         | 修改订单信息         |
| 抵赖（R）     | 审计     | 不承认做过某行为       | 不承认修改行为       |
| 信息泄露（I） | 保密性   | 信息被泄露或窃取       | 用户信息被泄露       |
| 拒绝服务（D） | 可用性   | 消耗资源、服务可不用   | DDOS 导致网站不可用  |
| 权限提升（E） | 授权     | 未经授权获取、提升权限 | 普通用户提升到管理员 |用户提升到管理员

**ASTRIDE** - 随着全球对隐私保护重视程度的加大，隐私安全也成了产品的一个重要威胁，因此 STRIDE 的 6 个威胁也添加了一项隐私（Privacy），也就变成了 ASTRIDE，A 代表 Advanced。

**DSA** - Digital Signature Algorithm 是Schnorr和ElGamal签名算法的变种，被美国NIST作为DSS(Digital Signature Standard)。简单的说，这是一种更高级的验证方式，用作数字签名。不单单只有公钥、私钥，还有数字签名。私钥加密生成数字签名，公钥验证数据及签名。如果数据和签名不匹配则认为验证失败。数字签名的作用就是校验数据在传输过程中不被修改。数字签名，是单向加密的升级。

> 签名中 R != 1

**DSA算法和RSA算法的异同：**

DSA算法是DSS技术的核心算法，与RSA算法的异同如下：

二者都是数字签名算法中的重要组成，缺一不可；
DSA算法仅仅包含数字签名算法，一般不用于加密算法中（某些扩展可以支持加密）；
DSA算法仅产生数字证书，信息仅能用于验证，不适合进行加密通信，所以HTTPS不会使用这个算法；
RSA算法包含加解密的密钥信息，同时可作为数字签名算法；

**IV** - 初始向量（Initialization Vector: IV）：为随机初始化向量，使用随机初始化向量才能达到语义安全。每次加密都需要随机生成一个不同的比特序列作为初始化向量(IV)，不可采用静态的IV值。还需要保证IV的随机生成是安全的。初始化向量的值，是根据密码算法不同。但最基本要求是“唯一性”，也就是同一个密钥不重复使用同一个初始化向量。

*随机数的安全性* - 密码学意义上的安全随机数，要求必须保证其随机性、不可预测性、不可重现性。

**RC4** -（来自Rivest Cipher 4的缩写）是一种流加密算法，密钥长度可变。它加解密使用相同的密钥，因此也属于对称加密算法。RC4是有线等效加密（WEP）中采用的加密算法，也曾经是TLS可采用的算法之一。

**Padding Oracle Attack**是比较早的一种漏洞利用方式，该漏洞主要是由于设计使用的场景不当，导致可以利用密码算法通过”旁路攻击“被破解，并不是对算法的破解。

利用该漏洞可以破解出密文的明文以及将明文加密成密文，该漏洞存在条件如下：

攻击者能够获取到密文（基于分组密码模式），以及IV向量（通常附带在密文前面，初始化向量）

攻击者能够修改密文触发解密过程，解密成功和解密失败存在差异性。

*分组密码* 是每次只能处理特定长度的一块数据的算法，每块都是一个分组，分组的比特数就称为分组长度，但是当加密的内容超过分组密码的分组长度时，就要对分组密码算法进行迭代，迭代的方法称为分组密码的模式。（ECB, CBC, CFB, OFB, CTR）


**OAEP**(RAS 的填充方式) - 如果要使用RSA算法，我们首先要将算法改造为非确定性，具体的做法就是通过“Padding”，就是将明文与某个随机数组合起来，这样，明文就被转移到了一个更大的明文空间上，一定程度上防止了选择明文攻击。但这样还是不够，随着选择密文攻击被提出，两千年前后，密码学界提出了更安全的Padding模式，即RSA-PSS和RSA-OAEP，后者正是PKCS#1 v2中使用RSA的标准方式。

**DES** - 全称为Data Encryption Standard，即数据加密标准，是一种使用密钥加密的块算法。设计中使用了分组密码设计的两个原则：混淆（confusion）和扩散(diffusion)，其目的是抗击敌手对密码系统的统计分析。混淆是使密文的统计特性与密钥的取值之间的关系尽可能复杂化，以使密钥和明文以及密文之间的依赖性对密码分析者来说是无法利用的。扩散的作用就是将每一位明文的影响尽可能迅速地作用到较多的输出密文位中，以便在大量的密文中消除明文的统计结构，并且使每一位密钥的影响尽可能迅速地扩展到较多的密文位中，以防对密钥进行逐段破译。

DES 算法的入口参数有三个：Key、Data、Mode。其中Key为7个字节共56位，是DES算法的工作密钥；Data为8个字节64位，是要被加密或被解密的数据；Mode为DES的工作方式，有两种：加密或解密。

经过16次迭代运算后，得到L16、R16,将此作为输入，进行逆置换，逆置换正好是初始置换的逆运算，由此即得到密文输出。
此算法是对称加密算法体系中的代表,在计算机网络系统中广泛使用。

**Kerberos** 是一种基于加密 Ticket 的身份认证协议。Kerberos 主要由三个部分组成：Key Distribution Center (即KDC)、Client 和 Service。客户端会先访问两次KDC，然后再访问目标Service，如：HTTP服务。




## Reference

DSA算法和RSA算法的异同 - https://www.jianshu.com/p/d5db48fe434b

IV - https://www.zhihu.com/question/559774307/answer/2717185761