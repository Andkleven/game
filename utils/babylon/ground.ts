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

export function ground(scene: Scene) {
  const groundSize = 350;
  const parkingLength = 30;
  const parkingWidth = 15;
  const numberParkingPlace = 20;

  const ground = MeshBuilder.CreateGround(
    "ground",
    { width: groundSize, height: groundSize },
    scene
  );
  const groundMaterial = new StandardMaterial("groundMaterial", scene);
  groundMaterial.diffuseColor = new Color3(0.75, 1, 0.25);
  ground.material = groundMaterial;
  ground.position.y = -1.5;
  for (let index = 0; index < numberParkingPlace; index++) {
    let startPoint = groundSize / 2 - 20 - index * parkingWidth;
    MeshBuilder.CreateLines(`line${index}`, {
      points: [
        new Vector3(0, 0, startPoint),
        new Vector3(parkingLength, 0, startPoint),
        new Vector3(parkingLength, 0, startPoint - parkingWidth),
      ],
      colors: [
        new Color4(1, 0, 0, 1),
        new Color4(1, 0, 0, 1),
        new Color4(1, 0, 0, 1),
      ],
    });
  }
  let startPoint = groundSize / 2 - 20 - numberParkingPlace * parkingWidth;
  MeshBuilder.CreateLines(`line${numberParkingPlace}`, {
    points: [
      new Vector3(0, 0, startPoint),
      new Vector3(parkingLength, 0, startPoint),
    ],
    colors: [new Color4(1, 0, 0, 1), new Color4(1, 0, 0, 1)],
  });
}
