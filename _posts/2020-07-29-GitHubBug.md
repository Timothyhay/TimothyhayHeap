---
layout: blogpage
title: 一种SublimeMerge不会推送到Github的commit
comments: true
tags: Git Skill
---

这个问题.. 居然被我碰到了两次！

{% highlight console %}
  layout: tagarchive
- title: Tag - Csharp
- tagname: Csharp
+ title: Tag - CSharp
+ tagname: CSharp
{% endhighlight %}

上面的变更记录只是修改文件名的时候同时修改的文件内容的记录。我把Csharp.md重命名为了CSharp.md。
就是说，在给文件名大小写做修改的时候，如果GitHub仓库里原始文件依然存在（这是当然的），那么对文件的修改会被保存，但是文件名的修改不会。
这会导致一个问题，本地的文件名已经是修改过了的，那么本地调试时如果需要使用超链接跳转或者引用这个文件，要求的路径自然也是改过的那个.. 
这种时候如果一并在使用GitHub Pages，网站上的链接就会指向错误的文件名，报404。

我觉得这个问题.. 其实还挺有意义？