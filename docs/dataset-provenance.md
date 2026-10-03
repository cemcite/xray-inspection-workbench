# Dataset provenance and usage gate

Checked on 2026-10-02.

The official PIDray repository describes 124,486 PNG X-ray images, 12 prohibited
item categories, manually created instance annotations, and official Google
Drive/Baidu download locations:

- https://github.com/lutao2021/PIDray
- https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Towards_Real-World_Prohibited_Item_Detection_A_Large-Scale_X-Ray_Benchmark_ICCV_2021_paper.html

The repository README exposes download links but does not state dataset usage
terms, and the repository does not expose an explicit license file. Public
availability is not itself permission to redistribute. Before downloading or
publishing derived samples, confirm the applicable terms with the dataset
maintainers or the terms accompanying the hosted files.

Consequences for this repository:

- No dataset or model weight is downloaded automatically.
- Raw and processed datasets remain ignored by Git.
- The converter accepts only a local COCO-format annotation file and image tree.
- Any future sample committed to `data/samples` must include source and license
  metadata proving redistribution is allowed.
