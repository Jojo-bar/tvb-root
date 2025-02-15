# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

import os
import numpy as np
import pandas as pd
import pytest
import siibra
from tvb.adapters.creators import siibra_base as sb
from tvb.adapters.creators.siibra_creator import CLB_AUTH_TOKEN_KEY
from tvb.datatypes import connectivity, graph
from tvb.tests.framework.core.base_testcase import BaseTestCase


def no_ebrains_auth_token():
    hbp_auth = os.environ.get(CLB_AUTH_TOKEN_KEY)
    return hbp_auth is None


HUMAN_ATLAS = 'Multilevel Human Atlas'
MONKEY_ATLAS = 'Monkey Atlas (pre-release)'
JULICH_PARCELLATION = 'Julich-Brain Cytoarchitectonic Maps 2.9'
MONKEY_PARCELLATION = 'Non-human primate'


@pytest.mark.skipif(no_ebrains_auth_token(), reason="No EBRAINS AUTH token for accesing the KG was provided!")
class TestSiibraBase(BaseTestCase):

    @pytest.fixture()
    def create_test_atlases_and_parcellations(self):
        self.human_atlas = siibra.atlases[HUMAN_ATLAS]
        self.monkey_atlas = siibra.atlases[MONKEY_ATLAS]
        self.julich_parcellation = siibra.parcellations[JULICH_PARCELLATION]
        self.monkey_parcellation = siibra.parcellations[MONKEY_PARCELLATION]

    @pytest.fixture()
    def create_weights_and_tracts(self, create_test_atlases_and_parcellations):
        """
        Return all the weights and tracts available in siibra for default atlas and parcellation
        """
        weights = siibra.get_features(self.julich_parcellation, siibra.modalities.StreamlineCounts)
        tracts = siibra.get_features(self.julich_parcellation, siibra.modalities.StreamlineLengths)
        self.weights = weights
        self.tracts = tracts

    @pytest.fixture()
    def create_siibra_functional_connectivities(self, create_test_atlases_and_parcellations):
        """
        Return all the functional connectivities available in siibra for default atlas and parcellation
        """
        fcs = siibra.get_features(self.julich_parcellation, siibra.modalities.FunctionalConnectivity)
        self.fcs = fcs

    def test_check_atlas_parcellation_compatible(self, create_test_atlases_and_parcellations):
        assert sb.check_atlas_parcellation_compatible(self.human_atlas, self.julich_parcellation)
        assert not sb.check_atlas_parcellation_compatible(self.monkey_atlas, self.julich_parcellation)

    def test_get_atlases_for_parcellation(self, create_test_atlases_and_parcellations):
        atlas_list = sb.get_atlases_for_parcellation(self.julich_parcellation)
        assert atlas_list
        assert self.human_atlas in atlas_list
        assert self.monkey_atlas not in atlas_list

    def test_get_parcellations_for_atlas(self, create_test_atlases_and_parcellations):
        parcellation_list = sb.get_parcellations_for_atlas(self.human_atlas)
        assert parcellation_list
        assert self.julich_parcellation in parcellation_list
        assert self.monkey_parcellation not in parcellation_list

    def test_parse_subject_ids(self):
        single_id = '000'
        assert sb.parse_subject_ids(single_id) == ['000']

        multiple_ids = '000;010'
        assert sb.parse_subject_ids(multiple_ids) == ['000', '010']

        range_ids = '000-002'
        assert sb.parse_subject_ids(range_ids) == ['000', '001', '002']

        range_and_multiple_ids = '000-002;010'
        assert sb.parse_subject_ids(range_and_multiple_ids) == ['000', '001', '002', '010']

        range_and_multiple_ids2 = '100;000-002;010'
        assert sb.parse_subject_ids(range_and_multiple_ids2) == ['100', '000', '001', '002', '010']

    def test_init_siibra_params_no_selections(self, create_test_atlases_and_parcellations):
        """"
        Test initialization of siibra paramas when no sellection was made for atlas, parcellation or subject ids
        """
        empty_params_config = sb.init_siibra_params(None, None, None)
        atlas, parcellation, subject_ids = empty_params_config
        assert atlas == self.human_atlas
        assert parcellation == self.julich_parcellation
        assert subject_ids == 'all'

    def test_init_siibra_params_atlas_selected(self, create_test_atlases_and_parcellations):
        """"
        Test initialization of siibra paramas when only the atlas was selected
        """
        empty_params_config = sb.init_siibra_params(self.human_atlas, None, None)
        _, parcellation, subject_ids = empty_params_config
        assert parcellation == self.julich_parcellation
        assert subject_ids == 'all'

    def test_init_siibra_params_parcellation_selected(self, create_test_atlases_and_parcellations):
        """"
        Test initialization of siibra paramas when only the parcellation was selected
        """
        empty_params_config = sb.init_siibra_params(None, self.julich_parcellation, None)
        atlas, _, subject_ids = empty_params_config
        assert atlas == self.human_atlas
        assert subject_ids == 'all'

    def test_init_siibra_params_subjects_selected(self, create_test_atlases_and_parcellations):
        """"
        Test initialization of siibra paramas when only the subjects were selected
        """
        empty_params_config = sb.init_siibra_params(None, None, '000;001')
        atlas, parcellation, subject_ids = empty_params_config
        assert atlas == self.human_atlas
        assert parcellation == self.julich_parcellation
        assert subject_ids == ['000', '001']

    def test_get_connectivity_component(self, create_test_atlases_and_parcellations):
        """
        Test the retrieval of structural connectivities (weights and tracts) and functional connectivities
        """
        weights = sb.get_connectivity_component(self.julich_parcellation, sb.Component2Modality.WEIGHTS)
        assert len(weights) > 0
        assert type(weights[0]) == siibra.features.connectivity.StreamlineCounts

        tracts = sb.get_connectivity_component(self.julich_parcellation, sb.Component2Modality.TRACTS)
        assert len(tracts) > 0
        assert type(tracts[0]) == siibra.features.connectivity.StreamlineLengths

        fcs = sb.get_connectivity_component(self.julich_parcellation, sb.Component2Modality.FC)
        assert len(fcs) > 0
        assert type(fcs[0]) == siibra.features.connectivity.FunctionalConnectivity

    def test_get_hemispheres_for_regions(self):
        reg_names = ['reg1_right', 'reg1_left', 'reg_2']
        hemi = sb.get_hemispheres_for_regions(reg_names)
        assert hemi == [1, 0, 0]

    def test_get_regions_positions(self, create_test_atlases_and_parcellations):
        region = self.human_atlas.get_region('v1', parcellation=self.julich_parcellation)
        reg_coord = sb.get_regions_positions([region])[0]
        assert reg_coord == (2.8424532907291535, -82.22873119424844, 2.1326498912705745)

    def test_filter_structural_connectivity_by_id(self, create_weights_and_tracts):
        subject = '000'
        filtered_weights, filtered_tracts = sb.filter_structural_connectivity_by_id(self.weights, self.tracts,
                                                                                    [subject])

        assert len(filtered_weights) == 1
        assert len(filtered_tracts) == 1

        assert filtered_weights[0].subject == subject
        assert filtered_tracts[0].subject == subject

    def test_filter_functional_connectivity_by_id(self, create_siibra_functional_connectivities):
        subject = '000'
        filtered_fcs = sb.filter_functional_connectivity_by_id(self.fcs, [subject])

        assert len(filtered_fcs) == 5
        for fc in filtered_fcs:
            assert fc.subject == subject

    def test_create_tvb_structural_connectivity(self):
        weights_data = np.random.randint(0, 5, size=(2, 2))
        tracts_data = np.random.randint(0, 5, size=(2, 2))
        regions = ['reg1', 'reg2']
        hemi = [1, 0]
        positions = [(2.8424532907291535, -82.22873119424844, 2.1326498912705745),
                     (4.8424532907291535, -52.22873119424844, 4.1326498912705745)]

        weights = pd.DataFrame(data=weights_data, index=regions, columns=regions)
        tracts = pd.DataFrame(data=tracts_data, index=regions, columns=regions)

        tvb_conn = sb.create_tvb_structural_connectivity(weights, tracts, regions, hemi, positions)

        assert (tvb_conn.region_labels == regions).all()
        assert tvb_conn.number_of_regions == 2
        assert (tvb_conn.centres == positions).all()
        assert (tvb_conn.hemispheres == hemi).all()

    def test_get_tvb_connectivities_from_kg(self, create_test_atlases_and_parcellations):
        tvb_conns = sb.get_tvb_connectivities_from_kg(self.human_atlas, self.julich_parcellation, '001')

        assert len(tvb_conns) == 1
        assert (list(tvb_conns.keys()) == ['001'])
        assert type(tvb_conns['001']) == connectivity.Connectivity

    def test_get_fc_name_from_file_path(self):
        path = 'c/users/user1/FunctionalConnectivity.Name.csv'
        name = 'FunctionalConnectivity.Name'

        assert sb.get_fc_name_from_file_path(path) == name

    def test_create_tvb_connectivity_measure(self, create_siibra_functional_connectivities):
        conn = connectivity.Connectivity.from_file("connectivity_192.zip")
        fc = self.fcs[0]

        # the FC and SC are not compatible, but are used together only for testing purposes
        tvb_conn_measure = sb.create_tvb_connectivity_measure(fc, conn)
        assert (tvb_conn_measure.array_data == fc.matrix.to_numpy()).all()
        assert tvb_conn_measure.connectivity is conn
        assert tvb_conn_measure.title == 'EmpCorrFC_concatenated'

    def test_get_connectivity_measures_from_kg(self, create_test_atlases_and_parcellations):
        sc1 = connectivity.Connectivity.from_file("connectivity_76.zip")
        scs = {'001': sc1}

        tvb_conn_measures = sb.get_connectivity_measures_from_kg(self.human_atlas, self.julich_parcellation, '001', scs)

        assert len(tvb_conn_measures) == 1
        assert len(tvb_conn_measures['001']) == 5
        assert (list(tvb_conn_measures.keys()) == ['001'])
        assert type(tvb_conn_measures['001'][0]) == graph.ConnectivityMeasure

        assert tvb_conn_measures['001'][0].connectivity is sc1

    def test_get_connectivities_from_kg_no_fc(self, create_test_atlases_and_parcellations):
        """
        Test retrieval of just structural connectivities
        """
        scs, fcs = sb.get_connectivities_from_kg(self.human_atlas, self.julich_parcellation, '001')

        assert len(scs) == 1
        assert not fcs

        assert (list(scs.keys()) == ['001'])
        assert type(scs['001']) == connectivity.Connectivity

    def test_get_connectivities_from_kg_with_fc(self, create_test_atlases_and_parcellations):
        """
        Test retrieval of both structural and functional connectivities
        """
        scs, fcs = sb.get_connectivities_from_kg(self.human_atlas, self.julich_parcellation, '001', True)

        assert len(scs) == 1
        assert len(fcs) == 1
        assert len(fcs['001']) == 5

        assert (list(scs.keys()) == ['001'])
        assert type(scs['001']) == connectivity.Connectivity

        assert (list(fcs.keys()) == ['001'])
        assert type(fcs['001'][4]) == graph.ConnectivityMeasure

