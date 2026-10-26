from fastapi import FastAPI


app = FastAPI()

products = []

class Product:
    def __init__(name, quantity):
        self.name = name
        self.quantity = quantity

@app.post("products/")
def add_product(name, quantity):
    prod = Product()

    prod.name = name 
    prod.quantity = quantity

    products.append(prod)
    return products 

