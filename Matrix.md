# Matrix:

```python
Matrix = [[1,0,0],[0,1,0],[0,0,1]]
```

```python
Matrix = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
]
```

| 1 | 0 | 0 |
| :---: | :---: | :---: |
|   0   |   1   |   0   |
|   0   |   0   |   1   |

---
# Matrix Schleife:

```python 
for y in range(len(Matrix) * len(Matrix[0])):
    Zeile = y // len(Matrix[0])
    Spalte = y % len(Matrix[0])
    Matrix[Zeile][Spalte] = Matrix[Zeile][Spalte] * n
```

- **Ziel:**
``` python
Matrix[Zeile][Spalte]
```

---
### for-schleife

```python
for y in range(len(Matrix) * len(Matrix[0])):
```


- Die Schleife läuft **y** mal von 0 bis zur Gesamtanzahl der Objekte in der Matrix.

- **len(Matrix)** ist die Anzahl der Objekte in der List, oder auch Zeilen, was hier 3 sind

| ***1*** |  0  |  0  |
| :---: | :---: | :---: |
| ***0*** |  1  |  0  |
| ***0*** |  0  |  1  |

- **len(Matrix[0])** ist die Anzahl der enthaltenen Objekte in der ersten Zeile

| ***1*** | ***0*** | ***0*** |
| :---: | :---: | :---: |
|   0   |   1   |   0   |
|   0   |   0   |   1   |

- also in diesem Fall eine **3x3 Matrix**
- 4x4 Matrix:

|  1  |  0  |  0  | 0   |
| :-: | :-: | :-: | --- |
|  0  |  1  |  0  | 0   |
|  0  |  0  |  1  | 0   |
|  0  |  0  |  0  | 1   |


---
### Zeile

```python
Zeile = y // len(Matrix[0])
```

- "Floor division"
- wird auf die nächste volle zahl abgerundet

```python
3/4 = 0.75
3//4 = 0

9/8 = 1.125
9//8 = 1
```

| y - Wert |      code       | Ergebniss |
| :------: | :-------------: | :-------: |
|  y = 0   | Zeile = 0 // 3  |     0     |
|  y = 1   | Zeile = 1 // 3  |     0     |
|  y = 2   | Zeile = 2 // 3  |     0     |
|  y = 3   | Zeile = 3 // 3  |     1     |
|  y = 4   | Zeile = 4 // 3  |     1     |

`Matrix[0][x]`
`[1,0,0]`


---
### Spalte

```python
Spalte = y % len(Matrix[0])
```

- "Modulus"
- ist nur der Rest einer Division
- wenn es weniger als einmal dividiert wurde gibt es das zu dividierende aus

```python
3/4 = 0.75
3%4 = 3

9/8 = 1.125
9%8 = 1
```

| y - Wert |      code      | Ergebniss |
| :------: | :------------: | :-------: |
|  y = 0   | Zeile = 0 % 3  |     0     |
|  y = 1   | Zeile = 1 % 3  |     1     |
|  y = 2   | Zeile = 2 % 3  |     2     |
|  y = 3   | Zeile = 3 % 3  |     0     |
|  y = 4   | Zeile = 4 % 3  |     1     |

`Matrix[0][0]`
`[1]`


---

## Durchlauf

| `y` | `Zeile (y // 3)` | `Spalte (y % 3)` | Matrix       |
| --- | ---------------- | ---------------- | ------------ |
| 0   | 0                | 0                | Matrix[0][0] |
| 1   | 0                | 1                | Matrix[0][1] |
| 2   | 0                | 2                | Matrix[0][2] |
| 3   | 1                | 0                | Matrix[1][0] |
| 4   | 1                | 1                | Matrix[1][1] |
| 5   | 1                | 2                | Matrix[1][2] |
| 6   | 2                | 0                | Matrix[2][0] |
| 7   | 2                | 1                | Matrix[2][1] |
| 8   | 2                | 2                | Matrix[2][2] |

---
