```python
Matrix = [[1,0,0],[0,1,0],[0,0,1]]

while True:
    try:
        n = int(input("Zahl:"))

        if n < 0 or n >= 10:
            print("Eingabe Zahl zuhoch oder niedrig")
            continue

        elif n == 0:
            print("Script beendet")
            break

        else:
            for y in range(len(Matrix) * len(Matrix[0])):
                Zeile = y // len(Matrix[0])
                Spalte = y % len(Matrix[0])
                Matrix[Zeile][Spalte] = Matrix[Zeile][Spalte] * n

            print(f"{Matrix[0]}\n{Matrix[1]}\n{Matrix[2]}")
            break

    except ValueError:
        print("Ungültige Eingabe")
```
---

# Ausgabe:

```python
Zahl:5
[5, 0, 0]
[0, 5, 0]
[0, 0, 5]
```
