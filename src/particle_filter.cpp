/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Modified on: May 2, 2017
 *      By: Morgane Lustman
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cfloat>
#include <map>

#include "particle_filter.h"
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. 
  num_particles = 1000;

  // Set x, y, theta / Add random Gaussian noise to each particle
  normal_distribution<double> N_x(x, std[0]);
  normal_distribution<double> N_y(y, std[1]);
  normal_distribution<double> N_theta(theta, std[2]);
  default_random_engine gen;

  particles.clear();
  particles.resize(num_particles);
  
  for (Particle &p : particles) {
    p.id = 0;
    p.x = N_x(gen);
    p.y = N_y(gen);
    p.theta = N_theta(gen);
    p.weight = 1;
  }

  // Initialize all weights to 1. 
  weights.clear();
  weights.resize(num_particles);
  for (int i = 0; i < weights.size(); i++) {
    weights[i] = 1.0;
  }

  // It is initialized
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  
  default_random_engine gen;

  // For each particle
  for (Particle &p : particles) {
    // move the particle
    if (fabs(yaw_rate) > 0.00001) {
      p.x += velocity/yaw_rate*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y += velocity/yaw_rate*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      p.theta += yaw_rate * delta_t;
    }
    else {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }

    // Add gaussian noise: Gaussian distribution with mean = updated particle position and std of measurement
    normal_distribution<double> N_x(p.x, std_pos[0]);
    normal_distribution<double> N_y(p.y, std_pos[1]);
    normal_distribution<double> N_theta(p.theta, std_pos[2]);

    p.x = N_x(gen);
    p.y = N_y(gen);
    p.theta = N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  // Go through all the observations
  for (int o = 0; o < observations.size(); o++) {
  
    LandmarkObs &obs = observations[o];

    // Store the minimum distance and closest landmark id to the observation so far.
    double min_d = DBL_MAX;
    int closest_landmark_id = 0;

    // Calculate the distance between this observation and the landmarks
    for (int l = 0; l < predicted.size(); l++) {
      LandmarkObs &landmark = predicted[l];
      double d = dist(obs.x, obs.y, landmark.x, landmark.y);
      
      if (d < min_d) {
        d = min_d;
        closest_landmark_id = landmark.id;
      }
    }

    // Assign closest landmark found to the observation
    obs.id = closest_landmark_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html

  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];

  // Iterate over all particles
  for (Particle &p : particles) {
    double part_x = p.x;
    double part_y = p.y;
    double part_theta = p.theta;

    // convert the observations into the map coordinates
    // stored in the observations_mcoord vector
    std::vector<LandmarkObs> observations_mcoord(observations);
    // Iterate over all observations
    for (LandmarkObs &obs : observations_mcoord) {
      double x2 = part_x + obs.x * cos(part_theta) - obs.y * sin(part_theta); 
      double y2 = part_y + obs.x * sin(part_theta) + obs.y * cos(part_theta);
      obs.x = x2;
      obs.y = y2;
      obs.id = 0;
    }

    // predicted vector will store all the landmarks within the sensor range
    std::vector<LandmarkObs> predicted;
    predicted.clear();

    LandmarkObs l;
    for (Map::single_landmark_s landmark : map_landmarks.landmark_list) {
      double d = dist(landmark.x_f, landmark.y_f, part_x, part_y);
      if (d <= sensor_range) {
        l.id = landmark.id_i;
        l.x = landmark.x_f;
        l.y = landmark.y_f;
        predicted.push_back(l);
      }
    }

    // Associate the closest predicted landmark to each observation
    dataAssociation(predicted, observations_mcoord);

    // Store landmark id and its information
    map<int,LandmarkObs> landmarkLookup;
    for (LandmarkObs p : predicted) {
      landmarkLookup.insert({p.id, p});
    }

    // Calculate new weight
    double weight = 1.0;
    // Iterate through all the observations_mcoord
    for (LandmarkObs observation : observations_mcoord) {
      double obs_x = observation.x;
      double obs_y = observation.y;
      double mu_x = landmarkLookup[observation.id].x;
      double mu_y = landmarkLookup[observation.id].y;

      double w1 = 1.0/(2.0*M_PI*sigma_x*sigma_y);
      double wx = pow((obs_x - mu_x), 2) / (2.0*pow(sigma_x, 2));
      double wy = pow((obs_y - mu_y), 2) / (2.0*pow(sigma_y, 2));
      double pxy = w1 * exp(-(wx + wy));

      // If probability is too low, assign it a minimum value 
      float _MINIMUM_PROBABILITY = 0.00001;
      if (pxy < _MINIMUM_PROBABILITY) {
        pxy = _MINIMUM_PROBABILITY;
      }

      weight *= pxy;

    }

    p.weight = weight;

  }

  // normalize the weights both in weights vector and in each particle's weight
  double sum_w = 0.0;
  for (Particle p: particles) {
    sum_w += p.weight;
  }

  weights.clear();

  for (Particle p: particles) {
    p.weight /= sum_w;
    weights.push_back(p.weight);
  }

}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::vector<Particle> particles_res;
  std::discrete_distribution<> d(weights.begin(), weights.end());
  default_random_engine gen;

 for (int i = 0; i<num_particles; i++) {
  int next_index = d(gen);
  particles_res.push_back(particles[next_index]);
 }

  particles = particles_res;

}

void ParticleFilter::write(std::string filename) {
  // You don't need to modify this file.
  std::ofstream dataFile;
  dataFile.open(filename, std::ios::app);
  for (int i = 0; i < num_particles; ++i) {
    dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
  }
  dataFile.close();
}