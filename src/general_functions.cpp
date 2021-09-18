/*
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
*/

#include "general_functions.h"

using namespace std;

// -----------------------------------
// -------- GENERAL FUNCTIONS --------
// -----------------------------------

// Split a string into a vector
vector<string> split(string s, char c) {
  string buff="";
  vector<string> v;

  for(unsigned int n=0; n<s.size(); n++) {
    if (s[n]!=c) buff+=s[n];
    else if (s[n]==c && buff!="") {
      v.push_back(buff);
      buff = "";
    }
  }
  if(buff!="") v.push_back(buff);

  return v;
}


// Save the contents of a matrix into a CSV file
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
void saveAsCSV(string name, Eigen::MatrixXd matrix) {
  ofstream file(name.c_str());
  file << matrix.format(CSVFormat);
}


// Convert the contents of a CSV file into a matrix
Eigen::MatrixXd getCSVcontent(string filename) {

  ifstream theCSV(filename);
  if(!theCSV.is_open()) throw runtime_error("Could not open "+filename+" file!");

  string line;

  // First pass to get size
  int nrows = 0;
  int ncols = 0;
  while(getline(theCSV, line)) {
    vector<string> strnums = split(line, ',');
    ncols = strnums.size();
    nrows++;
  }
  
  // Close-reopen file
  theCSV.clear();
  theCSV.seekg(0, ios_base::beg);
  
  Eigen::MatrixXd outmat(nrows, ncols);
  outmat.setZero(nrows, ncols);

  // Get content
  int lineID = 0;
  while(getline(theCSV, line)) {
    Eigen::VectorXd matline(ncols);

    vector<string> strnums = split(line, ',');
    for (int i = 0; i < strnums.size(); i++) {
      matline(i) = stod(strnums[i]);
    }
    
    outmat.row(lineID) << matline.transpose();

    lineID++;
  }
  
  return outmat;

}

