# Why GPS Data Isn't Stored as Strings in EXIF

## Question

If GPS coordinates are ultimately displayed as strings like `"37.422000°N, 122.084000°W"`, why doesn't Pillow just return them as strings? Why do we have to decode `IFDRational` fractions and convert bytes references?

## Answer: The EXIF/TIFF Specification

### GPS Data Structure in EXIF

GPS coordinates in EXIF are **not stored as strings** in the image file. They're stored as **binary rational numbers** (fractions) according to the TIFF 6.0 specification.

#### Example: What's Actually in the File

```text
GPSLatitude:
  - Tag ID: 2 (in GPSInfo IFD)
  - Type: RATIONAL (5)
  - Count: 3
  - Values: [37/1, 25/1, 15177/500]  # Three fractions (degrees, minutes, seconds)

GPSLatitudeRef:
  - Tag ID: 1
  - Type: ASCII (2)
  - Count: 2
  - Value: "N\0"  # Single character + null terminator
```

### Why Rational Numbers?

**Precision**: TIFF RATIONAL type = numerator (32-bit) / denominator (32-bit)

```python
Seconds: 15177/500 = 30.354 exactly
```

If stored as decimal string `"30.354"`, you lose information about the original fraction. If stored as `"30.35"`, you lose precision entirely.

**Flexibility**: Applications can choose their own precision when converting:

- Navigation app: 8 decimal places → `30.35400000°`
- Photo viewer: 2 decimal places → `30.35°`
- Surveying tool: Full rational → `15177/500`

**TIFF Specification Compliance**: EXIF is built on TIFF 6.0, which defines these exact data types. Cameras write TIFF-compliant binary data.

### What Pillow Returns

Pillow faithfully represents the binary EXIF data structure:

```python
from PIL.TiffImagePlugin import IFDRational

exif_data['GPSInfo'] = {
    1: b'N',                                    # GPSLatitudeRef (bytes, not string!)
    2: (IFDRational(37, 1),                     # GPSLatitude (rationals, not floats!)
        IFDRational(25, 1),
        IFDRational(15177, 500)),
    3: b'W',                                    # GPSLongitudeRef
    4: (IFDRational(122, 1),                    # GPSLongitude
        IFDRational(5, 1),
        IFDRational(24, 10))
}
```

**IFDRational** is a `numbers.Rational` subclass that:

- Preserves exact numerator/denominator from file
- Converts to float when needed: `float(IFDRational(15177, 500))` → `30.354`
- Converts to string: `str(IFDRational(15177, 500))` → `"30.354"`

**Bytes reference characters**: The cardinal directions (`N`, `S`, `E`, `W`) are stored as ASCII bytes with null terminators in the TIFF structure, not Python strings.

### Why We Need Conversion Code

Our code must:

1. **Extract rational tuples**: `(IFDRational, IFDRational, IFDRational)` → `(float, float, float)`
2. **Decode bytes**: `b'N'` → `'N'` (with error handling for malformed data)
3. **Convert DMS to decimal**: `(37°, 25', 30.354")` → `37.422567°`
4. **Format for display**: `37.422567, 'N'` → `"37.422567°N"`

### Example Conversion Flow

```python
# What's in the file (binary EXIF):
GPSLatitude = [0x00000025, 0x00000001,    # 37/1
               0x00000019, 0x00000001,    # 25/1
               0x00003B49, 0x000001F4]    # 15177/500

# What Pillow returns:
lat = (IFDRational(37, 1), IFDRational(25, 1), IFDRational(15177, 500))
lat_ref = b'N'

# What our code does:
def _convert_gps_coordinate(coord):
    return (float(coord[0]), float(coord[1]), float(coord[2]))
    # Returns: (37.0, 25.0, 30.354)

def dms_to_dd(dms, ref):
    deg, min_, sec = dms
    dd = deg + min_/60.0 + sec/3600.0
    # Returns: (37.422567, 'N')

# Final display:
"37.422567°N, 122.084000°W"
```

## Why Not Provide Convenience Methods?

Pillow **could** provide a method like:

```python
gps_string = exif.get_gps_coordinates_string()
# → "37.422567°N, 122.084000°W"
```

But this would:

1. **Force one format**: Not everyone wants `37.422567°N` format
   - Scientists might want signed decimals: `-37.422567`
   - Surveyors might want DMS: `37° 25' 30.354" N`
   - Some want 6 decimals, some want 8

2. **Lose precision**: Once converted to string, you can't get back the exact rational

3. **Hide data quality**: Raw rationals let you see if seconds = `0/1` (no GPS second data) vs `30354/1000` (precise measurement)

4. **Break TIFF philosophy**: Pillow is a low-level library that provides faithful access to the binary structure

## Best Practices

Our current code follows best practices:

1. **Preserve precision** through conversion chain
2. **Handle malformed data** (bytes that aren't ASCII, missing fields)
3. **Choose display format** appropriate for our use case (unsigned decimal + cardinal)
4. **Document conventions** so future maintainers understand the choices

## References

- [TIFF 6.0 Specification](https://www.adobe.io/open/standards/TIFF.html) - Section on RATIONAL type
- [EXIF 2.32 Specification](https://www.cipa.jp/std/documents/download_e.html?DC-008-Translation-2019-E) - GPS attribute tags
- [Pillow IFDRational source](https://github.com/python-pillow/Pillow/blob/main/src/PIL/TiffImagePlugin.py)
- [TIFF Tag 34853 (GPSInfo)](https://exiftool.org/TagNames/GPS.html)

## Conclusion

GPS data **is not** stored as strings in EXIF files—it's stored as binary rational numbers for maximum precision and flexibility. Pillow returns the data exactly as it appears in the file structure, and application code (like ours) decides how to format it for display. This design gives us control over precision, format, and error handling.

## Visual Data Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│ CAMERA: Takes photo with GPS receiver active                       │
│ GPS reading: 37° 25' 30.354" N, 122° 5' 2.4" W                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ EXIF WRITER: Stores in TIFF binary format                          │
│                                                                     │
│ GPSInfo IFD (Image File Directory):                                │
│   Tag 1 (GPSLatitudeRef):  ASCII "N\0"                             │
│   Tag 2 (GPSLatitude):     RATIONAL[3]                             │
│                            [37/1, 25/1, 15177/500]                 │
│   Tag 3 (GPSLongitudeRef): ASCII "W\0"                             │
│   Tag 4 (GPSLongitude):    RATIONAL[3]                             │
│                            [122/1, 5/1, 12/5]                      │
│                                                                     │
│ Binary (hex):                                                       │
│   00 00 00 25 00 00 00 01  (37/1)                                  │
│   00 00 00 19 00 00 00 01  (25/1)                                  │
│   00 00 3B 49 00 00 01 F4  (15177/500)                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PILLOW: Reads TIFF binary and creates Python objects               │
│                                                                     │
│ exif['GPSInfo'] = {                                                 │
│     1: b'N',                     # bytes (from ASCII)               │
│     2: (IFDRational(37, 1),      # fractions.Rational subclass     │
│         IFDRational(25, 1),                                         │
│         IFDRational(15177, 500)),                                   │
│     3: b'W',                                                        │
│     4: (IFDRational(122, 1),                                        │
│         IFDRational(5, 1),                                          │
│         IFDRational(12, 5))                                         │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ OUR CODE: Converts to floats and formats for display               │
│                                                                     │
│ _convert_gps_coordinate():                                          │
│   (IFDRational(37,1), ...) → (37.0, 25.0, 30.354)                  │
│                                                                     │
│ dms_to_dd():                                                        │
│   (37.0, 25.0, 30.354) → 37 + 25/60 + 30.354/3600 = 37.422567      │
│                                                                     │
│ Format string:                                                      │
│   f"{37.422567:.6f}°N" → "37.422567°N"                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: "37.422567°N, 122.084000°W"                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Insight

**The conversion chain is necessary because:**

1. **EXIF/TIFF specification** mandates binary rational number storage (not strings)
2. **Pillow** faithfully returns the exact binary structure as Python objects
3. **Applications** (like ours) choose precision and format for their specific needs

This separation of concerns is good design:

- File format (TIFF): Optimized for precision and storage
- Library (Pillow): Faithful low-level access
- Application (our code): Domain-specific formatting
