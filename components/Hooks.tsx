import SceneComponent from "babylonjs-hook";
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
} from "@babylonjs/core";
import { useAfterRender } from "react-babylonjs";
import * as CANNON from "cannon";
import { css } from "@emotion/react";
window.CANNON = CANNON;

export default function Hooks() {
  const onSceneReady = (scene: Scene) => {
    // This creates and positions a free camera (non-mesh)
    var camera = new FreeCamera("camera1", new Vector3(0, 29, 0), scene);

    // This targets the camera to scene origin
    camera.setTarget(Vector3.Zero());

    // This attaches the camera to the canvas
    camera.attachControl(scene, true);
    // lights
    new DirectionalLight("light1", new Vector3(0, 400, 0), scene);
    let light2 = new HemisphericLight("light2", new Vector3(0, 400, 0), scene);
    light2.intensity = 0.75;

    /***************************Car*********************************************/

    /*-----------------------Car Body------------------------------------------*/

    //Car Body Material
    let bodyMaterial = new StandardMaterial("body_mat", scene);
    bodyMaterial.diffuseColor = new Color3(1.0, 0.25, 0.25);
    bodyMaterial.backFaceCulling = false;
    bodyMaterial.diffuseTexture = new Texture(
      "http://i.imgur.com/ZUWbT6L.png",
      scene
    );

    //Array of points for trapezium side of car.
    let side = [
      new Vector3(-6.5, 1.5, -2),
      new Vector3(2.5, 1.5, -2),
      new Vector3(3.5, 0.5, -2),
      new Vector3(-9.5, 0.5, -2),
    ];

    side.push(side[0]); //close trapezium

    //Array of points for the extrusion path
    let extrudePath = [new Vector3(0, 0, 0), new Vector3(0, 0, 4)];

    //Create body and apply material
    let carBody = MeshBuilder.ExtrudeShape(
      "body",
      { shape: side, path: extrudePath, cap: Mesh.CAP_ALL },
      scene
    );
    carBody.material = bodyMaterial;
    // camera.parent = carBody;
    /*-----------------------End Car Body------------------------------------------*/

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

    /*************************** End Car*********************************************/

    /*****************************Add Ground********************************************/
    let groundSize = 400;

    let ground = MeshBuilder.CreateGround(
      "ground",
      { width: groundSize, height: groundSize },
      scene
    );
    let groundMaterial = new StandardMaterial("ground", scene);
    groundMaterial.diffuseColor = new Color3(0.75, 1, 0.25);
    ground.material = groundMaterial;
    ground.position.y = -1.5;
    /*****************************End Add Ground********************************************/

    /****************************Key Controls************************************************/

    let map = {}; //object for multiple key presses
    scene.actionManager = new ActionManager(scene);

    scene.actionManager.registerAction(
      new ExecuteCodeAction(ActionManager.OnKeyDownTrigger, function (evt) {
        map[evt.sourceEvent.key] = evt.sourceEvent.type == "keydown";
      })
    );

    scene.actionManager.registerAction(
      new ExecuteCodeAction(ActionManager.OnKeyUpTrigger, function (evt) {
        map[evt.sourceEvent.key] = evt.sourceEvent.type == "keydown";
      })
    );

    /****************************End Key Controls************************************************/

    /****************************Variables************************************************/

    let theta = 0;
    let deltaTheta = 0;
    let drive = 0; //distance translated per second
    let R = 50; //turning radius, initial set at pivot z value
    let NR; //Next turning radius on wheel turn
    let A = 4; // axel length
    let L = 4; //distance between wheel pivots
    let r = 1.5; // wheel radius
    let psi, psiRI, psiRO, psiFI, psiFO; //wheel rotations
    let phi; //rotation of car when turning

    let F; // frames per second

    /****************************End Variables************************************************/

    /****************************Animation******************************************************/

    scene.registerAfterRender(function () {
      F = scene.getEngine().getFps();
      if (drive > 0.15) {
        drive -= 0.15;
      } else if (drive < -0.15) {
        drive += 0.15;
      } else {
        drive = 0;
      }
      const distance = drive / F;
      psi = drive / (r * F);
      if ((map["a"] || map["A"]) && -Math.PI / 6 < theta) {
        deltaTheta = -Math.PI / 252;
        theta += deltaTheta;
        pivotFI.rotate(Axis.Y, deltaTheta, Space.LOCAL);
        pivotFO.rotate(Axis.Y, deltaTheta, Space.LOCAL);
        if (Math.abs(theta) > 0.00000001) {
          NR = A / 2 + L / Math.tan(theta);
        } else {
          theta = 0;
          NR = 0;
        }
        pivot.translate(Axis.Z, NR - R, Space.LOCAL);
        carBody.translate(Axis.Z, R - NR, Space.LOCAL);
        R = NR;
      }
      if ((map["d"] || map["D"]) && theta < Math.PI / 6) {
        deltaTheta = Math.PI / 252;
        theta += deltaTheta;
        pivotFI.rotate(Axis.Y, deltaTheta, Space.LOCAL);
        pivotFO.rotate(Axis.Y, deltaTheta, Space.LOCAL);
        if (Math.abs(theta) > 0.00000001) {
          NR = A / 2 + L / Math.tan(theta);
        } else {
          theta = 0;
          NR = 0;
        }
        pivot.translate(Axis.Z, NR - R, Space.LOCAL);
        carBody.translate(Axis.Z, R - NR, Space.LOCAL);
        R = NR;
      }
      if (map["w"] || map["W"]) {
        drive = 10;
      }
      if (map["s"] || map["S"]) {
        drive = -10;
      }
      if (Math.abs(drive) > 0) {
        phi = drive / (R * F);
        if (Math.abs(theta) > 0) {
          pivot.rotate(Axis.Y, phi, Space.WORLD);
          psiRI = drive / (r * F);
          psiRO = (drive * (R + A)) / (r * F);
          psiFI = (drive * Math.sqrt(R * R + L * L)) / (r * F);
          psiFO = (drive * Math.sqrt((R + A) * (R + A) + L * L)) / (r * F);
          wheelFI.rotate(Axis.Y, psiFI, Space.LOCAL);
          wheelFO.rotate(Axis.Y, psiFO, Space.LOCAL);
          wheelRI.rotate(Axis.Y, psiRI, Space.LOCAL);
          wheelRO.rotate(Axis.Y, psiRO, Space.LOCAL);
        } else {
          pivot.translate(Axis.X, -distance, Space.LOCAL);
          wheelFI.rotate(Axis.Y, psi, Space.LOCAL);
          wheelFO.rotate(Axis.Y, psi, Space.LOCAL);
          wheelRI.rotate(Axis.Y, psi, Space.LOCAL);
          wheelRO.rotate(Axis.Y, psi, Space.LOCAL);
        }
      }
    });
  };

  return (
    <SceneComponent
      antialias
      onSceneReady={onSceneReady}
      id="my-canvas"
      css={css`
        width: 100%;
        height: 100%;
      `}
    />
  );
}
