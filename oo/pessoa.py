class Pessoa:
    def __init__(self, *filhos nome = None, idade=35):
        self.idade = idade
        self.nome = nome
        self.filhos = list(filhos)

    def cumprimentar(self):
        return f'Olá {id(self)}'


if __name__== 'main':
    p = Pessoa('Luciano')
    print(pessoa.cumprimentar(p))
    print(id(p))
    print(p.cumprimentar())
    print(p.nome)
    p.nome = 'Renzo'
    print(p.nome)
    print(p.idade)