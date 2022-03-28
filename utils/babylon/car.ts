import {
  Vector3,
  Axis,
  Space,
  Color3,
  ExecuteCodeAction,
  Mesh,
  Material,
  Vector4,
  BaseTexture,
  Color4,
  ActionManager,
  ArcRotateCamera,
  FreeCamera,
  DirectionalLight,
  HemisphericLight,
  Scene,
  StandardMaterial,
  MeshBuilder,
  Texture,
  SolidParticleSystem,
  InstancedMesh,
} from "@babylonjs/core";

export function createCar(
  scene: Scene,
  {
    side = [
      new Vector3(-6.5, 1.5, -2),
      new Vector3(2.5, 1.5, -2),
      new Vector3(3.5, 0.5, -2),
      new Vector3(-9.5, 0.5, -2),
    ],
    extrudePath = [new Vector3(0, 0, 0), new Vector3(0, 0, 4)],
    diffuseColor = new Color3(1.0, 0.25, 0.25),
    image = "http://i.imgur.com/ZUWbT6L.png",
  }: {
    side?: [Vector3, Vector3, Vector3, Vector3];
    extrudePath?: [Vector3, Vector3];
    diffuseColor?: Color3;
    image?: string;
  }
): [
  pivotFI: Mesh,
  pivotFO: Mesh,
  pivot: Mesh,
  carBody: Mesh,
  wheelFI: Mesh,
  wheelFO: InstancedMesh,
  wheelRI: InstancedMesh,
  wheelRO: InstancedMesh
] {
  /*-----------------------Car Body------------------------------------------*/
  //Car Body Material
  let bodyMaterial = new StandardMaterial("body_mat", scene);
  bodyMaterial.diffuseColor = diffuseColor;
  bodyMaterial.backFaceCulling = false;
  bodyMaterial.diffuseTexture = new Texture(image, scene);

  side.push(side[0]); //close trapezium

  //Create body and apply material
  let carBody = MeshBuilder.ExtrudeShape(
    "body",
    { shape: side, path: extrudePath, cap: Mesh.CAP_ALL },
    scene
  );
  carBody.material = bodyMaterial;
  /*-----------------------Ensd Car Body------------------------------------------*/

  /*-----------------------Wheel------------------------------------------*/

  //Wheel Material
  let wheelMaterial = new StandardMaterial("wheel_mat", scene);
  let wheelTexture = new Texture("http://i.imgur.com/ZUWbT6L.png", scene);
  wheelMaterial.diffuseTexture = wheelTexture;

  //Set color for wheel tread as black
  let faceColors: Color4[] = [];
  faceColors[1] = new Color4(0, 0, 0);

  //set texture for flat face of wheel
  let faceUV: Vector4[] = [];
  faceUV[0] = new Vector4(0, 0, 1, 1);
  faceUV[2] = new Vector4(0, 0, 1, 1);

  //create wheel front inside and apply material
  let wheelFI = MeshBuilder.CreateCylinder(
    "wheelFI",
    {
      diameter: 3,
      height: 1,
      tessellation: 24,
      faceColors,
      faceUV: faceUV,
    },
    scene
  );
  wheelFI.material = wheelMaterial;

  //rotate wheel so tread in xz plane
  wheelFI.rotate(Axis.X, Math.PI / 2, Space.WORLD);
  /*-----------------------End Wheel------------------------------------------*/

  /*-------------------Pivots for Front Wheels-----------------------------------*/
  let pivotFI = new Mesh("pivotFI", scene);
  pivotFI.parent = carBody;
  pivotFI.position = new Vector3(-6.5, 0, -1);

  let pivotFO = new Mesh("pivotFO", scene);
  pivotFO.parent = carBody;
  pivotFO.position = new Vector3(-6.5, 0, 1);
  /*----------------End Pivots for Front Wheels--------------------------------*/

  /*-------------------Pivots for Back Wheels-----------------------------------*/
  let pivotRI = new Mesh("pivotRI", scene);
  pivotRI.parent = carBody;
  pivotRI.position = new Vector3(1, 0, -1);

  let pivotRO = new Mesh("pivotRO", scene);
  pivotRO.parent = carBody;
  pivotRO.position = new Vector3(1, 0, 1);
  /*----------------End Pivots for Back Wheels--------------------------------*/

  /*------------Create other Wheels as Instances, Parent and Position----------*/
  let wheelFO = wheelFI.createInstance("FO");
  wheelFO.parent = pivotFO;
  wheelFO.position = new Vector3(0, 0, 1.8);

  let wheelRI = wheelFI.createInstance("RI");
  wheelRI.parent = pivotRI;
  wheelRI.position = new Vector3(0, 0, -1.8);

  let wheelRO = wheelFI.createInstance("RO");
  wheelRO.parent = pivotRO;
  wheelRO.position = new Vector3(0, 0, 1.8);

  wheelFI.parent = pivotFI;
  wheelFI.position = new Vector3(0, 0, -1.8);
  /*------------End Create other Wheels as Instances, Parent and Position----------*/

  /*---------------------Create Car Centre of Rotation-----------------------------*/
  const pivot = new Mesh("pivot", scene); //current centre of rotation
  pivot.position.z = 50;
  carBody.parent = pivot;
  carBody.position = new Vector3(0, 0, -50);

  /*---------------------End Create Car Centre of Rotation-------------------------*/

  return [pivotFI, pivotFO, pivot, carBody, wheelFI, wheelFO, wheelRI, wheelRO];
}
