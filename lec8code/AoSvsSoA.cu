// AoS (Array of Structures): Data is organized as an array where each element is a structure containing multiple fields.
#define N 200

struct Particle {
    float x;
    float y;
    float z;
};

struct Particle particles[N]; // Array of structures

// particles[i].x = 1.0f;
// particles[i].y = 2.0f;
// particles[i].z = 3.0f;


// SoA (Structure of Arrays): Data is organized as separate arrays for each field or member of a structure.
 
float x[N], y[N], z[N]; // Arrays for x, y, z coordinates

// Access example
// x[i] = 1.0f;
// y[i] = 2.0f;
// z[i] = 3.0f;
