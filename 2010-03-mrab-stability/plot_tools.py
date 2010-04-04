import numpy

def group_by_first(cursor):
    last = None
    group = []
    for row in cursor:
        if last is None:
            last = row[0]
        elif last != row[0]:
            yield last, group
            group = []
            last = row[0]

        group.append(row[1:])

    if group:
        yield row[0], group

def auto_xy_reshape(cursor):
    x_values = []
    y_values = []
    z_values = []

    for x, y, v in cursor:
        z_values.append(v)
        if x not in x_values:
            x_values.append(x)

        if y not in y_values:
            y_values.append(y)

    z_values = numpy.array(z_values).reshape(
            (len(x_values), len(y_values)), order="C")

    return x_values, y_values, z_values

def unwrap_list(iterable):
    return list(row[0] for row in iterable)
