# x = [(j*i for j in range(1,i)])for i in range(1,10)]
y = [[(x*y)for y in range(1,x+1)]for x in range(1,10)]
print(y)
# y = [1,2]
# print(x,y)
# print('\n'.join([''.join(['%s*%s=%-2s '%(y,x,x*y)for y in range(1,x+1)])for x in range(1,10)]))

# l = '*'.join(["1","2"])
# print(l)

# python列表解析创建新列表的模板
# new = [基元 for i in range()] 其中基元可以是任意复杂的形式，但相对与“for i in range()” 来说，基元必须看作一个整体，不可涉及两个对象

# 该形式正确 y = [([],[]) for x in range(1,2)]
# 虽然[],[]外面相差一个括号，但该形式是错误的 y = [[],[] for x in range(1,2)]