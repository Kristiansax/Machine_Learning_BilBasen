# to_mean = []

# for price in basen.price:
#     if price is not '':
#         to_mean.append(price)

# price_mean = int(sum(to_mean))/sum(len(to_mean))

# basen.price = [price.replace('', price_mean) for price in basen.price]




# basen['price'] = basen['price'].str.strip()
# basen['price'] = basen['price'].map(lambda x: str(x)[:-3])
# basen['price'] = basen['price'].str.replace('.', '')




# for s in basen.name:
#     s = s.replace('.', ' ')
#     ss = s.split(' ')

#     if ss[0] is 'Citroxebn':
#         manufacturer.append('Citroen')
#     else:
#         manufacturer.append(ss[0])

#     if ss[1] is 'p!':
#        model.append('Up!')
#     else:
#         model.append(ss[1])

#     capacity.append(ss[2])