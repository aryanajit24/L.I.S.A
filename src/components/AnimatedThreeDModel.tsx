import { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF, PerspectiveCamera, Float, Environment } from '@react-three/drei';
import { Mesh, Group, MathUtils } from 'three';

function Model() {
  const group = useRef<Group>(null);
  
  // Simple 3D objects instead of loading GLTF models
  useFrame((state) => {
    if (!group.current) return;
    
    // Rotate the group slowly
    group.current.rotation.y = MathUtils.lerp(
      group.current.rotation.y,
      state.mouse.x * Math.PI * 0.1,
      0.05
    );
    
    group.current.rotation.x = MathUtils.lerp(
      group.current.rotation.x,
      state.mouse.y * Math.PI * 0.05,
      0.05
    );
  });

  return (
    <group ref={group}>
      <Float
        speed={2}
        rotationIntensity={0.4}
        floatIntensity={0.4}
      >
        {/* Central sphere */}
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[1.2, 32, 32]} />
          <meshStandardMaterial
            color="#0284c7"
            emissive="#0284c7"
            emissiveIntensity={0.4}
            roughness={0.2}
            metalness={0.8}
          />
        </mesh>
        
        {/* Orbiting torus */}
        <mesh position={[0, 0, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[2.5, 0.1, 16, 100]} />
          <meshStandardMaterial
            color="#d946ef"
            emissive="#d946ef"
            emissiveIntensity={0.2}
            roughness={0.4}
            metalness={0.6}
            transparent
            opacity={0.7}
          />
        </mesh>
        
        {/* Orbiting smaller spheres */}
        <mesh position={[2.5, 0, 0]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial
            color="#d946ef"
            emissive="#d946ef"
            emissiveIntensity={0.5}
            roughness={0.2}
            metalness={0.8}
          />
        </mesh>
        
        <mesh position={[-1.8, 1.8, 0]}>
          <sphereGeometry args={[0.25, 16, 16]} />
          <meshStandardMaterial
            color="#7dd3fc"
            emissive="#7dd3fc"
            emissiveIntensity={0.5}
            roughness={0.2}
            metalness={0.8}
          />
        </mesh>
        
        <mesh position={[-1.8, -1.8, 0]}>
          <sphereGeometry args={[0.2, 16, 16]} />
          <meshStandardMaterial
            color="#ffffff"
            emissive="#ffffff"
            emissiveIntensity={0.5}
            roughness={0.2}
            metalness={0.8}
          />
        </mesh>
      </Float>
    </group>
  );
}

function AnimatedThreeDModel() {
  return (
    <Canvas>
      <PerspectiveCamera makeDefault position={[0, 0, 10]} fov={50} />
      <ambientLight intensity={0.2} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} color="#d946ef" />
      <Model />
      <Environment preset="city" />
    </Canvas>
  );
}

export default AnimatedThreeDModel;