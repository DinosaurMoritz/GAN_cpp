from PIL import Image

with open("pic.txt") as f:
    data = f.readlines()
    data = [int(float(n)*255) for n in data[0].split(",") if n]


    def turn_to_pic(data=(), type="L", size=(28, 28), name="pic.png"):
        im = Image.new(type, size)
        im.putdata(data)
        im.save(name)


    turn_to_pic(data)


"""
with open("pic1.txt") as f:
    data = f.readlines()
    data = [int(float(n)*255) for n in data[0].split(",") if n]


    def turn_to_pic(data=(), type="L", size=(28, 28), name="pic1.png"):
        im = Image.new(type, size)
        im.putdata(data)
        im.save(name)


    turn_to_pic(data)

with open("pic2.txt") as f:
    data = f.readlines()
    data = [int(float(n)*255) for n in data[0].split(",") if n]


    def turn_to_pic(data=(), type="L", size=(28, 28), name="pic2.png"):
        im = Image.new(type, size)
        im.putdata(data)
        im.save(name)


    turn_to_pic(data)
"""