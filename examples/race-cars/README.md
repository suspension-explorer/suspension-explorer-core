# Race-car example layouts

Runnable, illustrative double-wishbone axles informed by published race-car
architectures. Hardpoints are authored examples, not measured team data.

## Examples

- [`fsae_pushrod.yaml`](fsae_pushrod.yaml): FSAE / FSUK-Style pushrod,
  horizontal rockers, inboard coilovers, T-bar ARB, heave link.
- [`fsae_pullrod.yaml`](fsae_pullrod.yaml): FSAE / FSUK-Style pullrod,
  near-vertical rockers, inboard coilovers, T-bar ARB, heave link.
- [`indycar_pushrod.yaml`](indycar_pushrod.yaml): IndyCar-Style pushrod,
  near-vertical rockers, inboard coilovers, U-bar ARB, heave link.
- [`formula_one_pushrod.yaml`](formula_one_pushrod.yaml): F1-Style pushrod,
  horizontal rockers, torsion bars and dampers, U-bar ARB, heave link.
- [`formula_one_pullrod.yaml`](formula_one_pullrod.yaml): F1-Style pullrod,
  horizontal rockers, torsion bars and dampers, T-bar ARB, heave link.

## References

Reviewed September 2026.

- [Formula Student Germany 2019 team specifications](https://www.formulastudent.de/fileadmin/user_upload/all/media/magazine/FSG2019_magazine_v20190724_LQ.pdf)
- [Research on anti-roll-bar design and testing for Formula Student](https://link.springer.com/article/10.1007/s42452-021-04279-z)
- [Design of a Formula Student Race Car Spring-Damper System, TU Eindhoven CST2010.024](https://www.f1-forecast.com/pdf/F1-Files/Design%20of%20a%20Formula%20Student%20Race%20Car%20Spring-Damper%20System.pdf)
- [McLaren MCL35 technical specification](https://www.mclaren.com/racing/formula-1/2020/car-launch/mclaren-mcl35-technical-specification/)
- [McLaren MCL60 technical specification](https://www.mclaren.com/racing/formula-1/2023/car-launch/mclaren-mcl60-technical-specification/)
- [McLaren MCL38 technical specification](https://www.mclaren.com/racing/formula-1/2024/mclaren-mcl38-technical-specification/)
- [iRacing Dallara IR18 manual](https://s100.iracing.com/wp-content/uploads/2023/10/Dallara-IR18-Manual.pdf)
- [INDYCAR: new car coming to iRacing](https://www.indycar.com/news/2018/03/03-02-new-car-coming-to-iracing)
- [OptimumG, Springs & Dampers, Part Three](https://optimumg.com/wp-content/uploads/2020/01/SpringsDampers_Tech_Tip_3.pdf)

## Run an example

From the core repository:

```sh
uv run --extra cli kinematics sweep \
  --geometry examples/race-cars/fsae_pushrod.yaml \
  --sweep examples/race-cars/axle-bump.yaml \
  --out /tmp/fsae-bump.csv
```

`axle-bump.yaml` drives both wheels together through -25 to +25 mm with the rack
held centered. `axle-roll.yaml` drives equal and opposite travel.

## Export to the web app

`catalog.json` lists the examples for the web app. After changing a geometry or
the catalog, regenerate the app's bundled copy:

```sh
uv run --extra cli python scripts/export_race_examples.py \
  ../app/backend/src/app/race_examples.json
```
