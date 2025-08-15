import subprocess
import json

def convert_to_mmacs(input_string):
  """
  Wandelt eine Zeichenkette mit MMACs oder GMACs in einen Float-Wert von MACs um.

  Args:
    input_string: Die Eingabezeichenkette (z.B. '112.35 MMACs' oder '4.46 GMACs').

  Returns:
    Ein Float-Wert, der die Anzahl der MACs darstellt, oder None bei einem Fehler.
  """
  try:
    # Trennt die Zeichenkette in den numerischen Teil und die Einheit
    parts = input_string.split()
    value_str = parts[0]
    unit = parts[1].upper()

    # Wandelt den numerischen Teil in einen Float um
    value = float(value_str)

    # Wendet den entsprechenden Multiplikator an
    if unit == 'MMACS':
      return value
    elif unit == 'GMACS':
      return value * 1_000
    else:
      print(f"Fehler: Unbekannte Einheit '{parts[1]}'")
      return None
  except (ValueError, IndexError):
    print(f"Fehler: Ungültiges Eingabeformat '{input_string}'")
    return None

def get_mmacs(p_list):
    print()
    p_json = json.dumps(p_list)

    # Subprozess-Aufruf
    result = subprocess.run(
        [
            '/home/stu242207/HydraViT/venv/bin/python',
            '/home/stu242207/HydraViT/validate_macs.py',  # oder wie dein validierungsskript heißt
            '/data22/datasets/ImageNet100/val',
            '--model', 'convnext_tiny',
            '--num-classes', '1000 ',
            '-b', '1',
            '--p-list', p_json
        ],
        capture_output=True,
        text=True
    )

    # Ergebnis verarbeiten
    try:
        output = result.stdout.strip().splitlines()[-1]  # Nimm letzte Zeile
        data = json.loads(output)
        return convert_to_mmacs(data['macs'])
    except Exception as e:
        print("Fehler beim Parsen:", e)
        print("stdout war:", result.stdout)
        print("stderr war:", result.stderr)

# Deine p_list
#p_list = [0.01] * 100
#print(get_mmacs(p_list))

# lists = [
#     # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#     # [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
#     # [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
#     # [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
#     #[0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125]
#     #[0.14999999999999925, 0.14999999999999925, 0.14999999999999925, 0.4599999999999995, 0.4599999999999995, 0.4599999999999995, 0.8899999999999999, 0.8399999999999999, 0.8399999999999999, 0.8299999999999998, 0.8599999999999999, 0.8599999999999999, 0.27999999999999936, 0.23999999999999932, 0.23999999999999932, 0.6999999999999997, 0.28999999999999937, 0.28999999999999937]
#     [0.07999999999999925, 0.07999999999999925, 0.07999999999999925, 0.17999999999999927, 0.17999999999999927, 0.23999999999999932, 0.12999999999999923, 0.13999999999999924, 0.12999999999999923, 0.13999999999999924, 0.13999999999999924, 0.12999999999999923, 0.07999999999999925, 0.06999999999999926, 0.06999999999999926, 0.2099999999999993, 0.07999999999999925, 0.07999999999999925]
# ]

# for l in lists:
#    print(f"{l}:")
#    print(get_mmacs(l))
#    print("\n")